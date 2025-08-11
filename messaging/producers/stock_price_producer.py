# messaging/producers/stock_price_producer.py

import json
import time
import threading
from datetime import datetime, timezone
from typing import Set, List

import pandas as pd
import yfinance as yf
from kafka import KafkaProducer, KafkaConsumer

from config.kafka_config import KAFKA_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)


# --------- Config ---------
BASE_COUNT = 100                 # noyau initial S&P500 à suivre
BATCH_SIZE = 40                  # taille des paquets pour yfinance
SLEEP_BETWEEN_BATCHES = 1.0      # secondes entre paquets
LOOP_SLEEP = 10                  # secondes entre deux boucles d’envoi
NEWS_TOPIC = KAFKA_CONFIG["topics"]["news_sentiment"]
PRICES_TOPIC = KAFKA_CONFIG["topics"]["stock_prices"]
CACHE_PATH = "/home/saadyaq/SE/Python/finsentbot/data/raw/seen_symbols.json"
PRICES_JSONL = "/home/saadyaq/SE/Python/finsentbot/data/raw/stock_prices.jsonl"
TOLERATE_EMPTY_HISTORY = False   # si True, on ignore les symboles sans historique du jour

# --------- Utils ---------
def get_sp500_symbols() -> List[str]:
    try:
        logger.info("Fetching S&P 500 symbols from Wikipedia")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df = pd.read_html(url)[0]
        syms = df["Symbol"].astype(str).str.upper().str.strip().tolist()
        # yfinance préfère des tickers sans espaces
        symbols = [s.replace(".", "-") for s in syms]
        logger.info(f"Successfully fetched {len(symbols)} S&P 500 symbols")
        return symbols
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 symbols: {e}")
        return []

def load_seen_symbols() -> Set[str]:
    try:
        with open(CACHE_PATH, "r") as f:
            data = json.load(f)
            symbols = set(str(s).upper().strip() for s in data if s)
            logger.debug(f"Loaded {len(symbols)} cached symbols")
            return symbols
    except FileNotFoundError:
        logger.info("No cached symbols file found, starting fresh")
        return set()
    except Exception as e:
        logger.error(f"Failed to load cached symbols: {e}")
        return set()

def save_seen_symbols(symbols: Set[str]) -> None:
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(sorted(list(symbols)), f)
        logger.debug(f"Saved {len(symbols)} symbols to cache")
    except Exception as e:
        logger.error(f"Could not save seen symbols: {e}")

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]


# --------- Kafka ---------
producer = KafkaProducer(
    bootstrap_servers=KAFKA_CONFIG["bootstrap_servers"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    **KAFKA_CONFIG["producer_config"]
)

# Set partagé et thread-safe (protégé par un lock)
TRACKED = set()
LOCK = threading.Lock()


def watch_news_symbols():
    """
    Thread Kafka Consumer qui écoute news_sentiment et ajoute à chaud
    les symboles détectés dans les articles.
    """
    consumer = KafkaConsumer(
        NEWS_TOPIC,
        bootstrap_servers=KAFKA_CONFIG["bootstrap_servers"],
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    print("👂 Watching news_sentiment for new symbols…")
    added_since_save = 0
    while True:
        try:
            for msg in consumer:
                try:
                    symbol = str(msg.value.get("symbol", "")).upper().strip()
                    if not symbol or symbol in {"NAN", "NONE", "NULL"}:
                        continue
                    with LOCK:
                        if symbol not in TRACKED:
                            TRACKED.add(symbol)
                            added_since_save += 1
                            print(f"➕ Will track new symbol from news: {symbol}")
                    # Sauvegarder périodiquement
                    if added_since_save >= 10:
                        with LOCK:
                            save_seen_symbols(TRACKED)
                            added_since_save = 0
                except Exception as e:
                    print(f"⚠️ Error processing news message: {e}")
        except Exception as e:
            print(f"⚠️ news consumer error: {e}; retrying in 5s")
            time.sleep(5)


def init_tracked():
    base = get_sp500_symbols()[:BASE_COUNT]
    seen = load_seen_symbols()
    with LOCK:
        TRACKED.clear()
        TRACKED.update(base)
        TRACKED.update(seen)
    print(f"🔧 Initial TRACKED size: {len(TRACKED)} "
          f"(base={len(base)}, seen_cache={len(seen)})")


def fetch_and_send_prices():
    """
    Boucle principale : prend un snapshot de TRACKED, fetch par paquets,
    envoie les derniers prix sur Kafka + JSONL.
    """
    while True:
        try:
            with LOCK:
                symbols_list = sorted(list(TRACKED))

            if not symbols_list:
                print("⚠️ No symbols to track. Sleeping…")
                time.sleep(LOOP_SLEEP)
                continue

            print(f"🔍 Fetching stock prices for {len(symbols_list)} symbols")
            for group in chunked(symbols_list, BATCH_SIZE):
                try:
                    # yfinance accepte une chaîne séparée par espaces ou une liste
                    tickers = yf.Tickers(" ".join(group))
                    for sym in group:
                        try:
                            yf_t = tickers.tickers.get(sym)
                            if yf_t is None:
                                print(f"⚠️ yfinance: unknown ticker {sym}")
                                continue

                            # Essai 1: 1d/1m
                            data = yf_t.history(period="1d", interval="1m")
                            if data.empty:
                                # Essai 2: 5d/5m
                                data = yf_t.history(period="5d", interval="5m")

                            if data.empty:
                                if not TOLERATE_EMPTY_HISTORY:
                                    print(f"⚠️ No data for {sym}")
                                continue

                            current_price = float(data["Close"].iloc[-1])
                            message = {
                                "symbol": sym,
                                "price": round(current_price, 2),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                            producer.send(PRICES_TOPIC, value=message)
                            # persistance locale
                            with open(PRICES_JSONL, "a") as f:
                                f.write(json.dumps(message) + "\n")
                        except Exception as e_sym:
                            print(f"⚠️ Error fetching {sym}: {e_sym}")
                    time.sleep(SLEEP_BETWEEN_BATCHES)
                except Exception as e_batch:
                    print(f"⚠️ Batch error: {e_batch}. Continuing…")

            print(f"✅ Sleeping {LOOP_SLEEP} seconds…\n")
            time.sleep(LOOP_SLEEP)

        except Exception as e_loop:
            print(f"⚠️ Loop error: {e_loop}. Backing off 5s…")
            time.sleep(5)


if __name__ == "__main__":
    # Init liste de tickers (base + cache)
    init_tracked()

    # Thread d’écoute des symboles vus dans les news
    t = threading.Thread(target=watch_news_symbols, daemon=True)
    t.start()

    # Boucle principale de fetch des prix
    fetch_and_send_prices()
