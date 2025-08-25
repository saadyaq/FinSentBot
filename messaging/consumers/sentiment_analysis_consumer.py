import json
from kafka import KafkaConsumer, KafkaProducer
import pandas as pd
import requests
from config.kafka_config import KAFKA_CONFIG
import re
from nlp.preprocessing import clean_text
from nlp.sentiment_model import SentimentModel
from utils.logger import setup_logger

logger = setup_logger(__name__)

def get_symbol_map() -> dict:
    """
    Construit un dictionnaire qui associe différents alias (ticker, nom complet,
    variantes sans suffixes) à leur ticker officiel du S&P 500. On récupère la
    liste des entreprises sur Wikipédia et on ajoute quelques alias manuels
    supplémentaires (Google, Facebook, Tesla, etc.) pour améliorer la détection.
    """
    try:
        logger.info("Building symbol map from S&P 500 companies")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        df = pd.read_html(response.text)[0]
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 data for symbol mapping: {e}")
        return {}

    symbol_map: dict[str, str] = {}
    # Regex pour supprimer les suffixes courants (INC., CORP., CO., etc.)
    suffix_re = re.compile(
        r"\b(?:INCORPORATED|INC\.?|CORPORATION|CORP\.?|COMPANY|CO\.?|LP|L\.P\.|HOLDING(S)?|GROUP)\b"
    )

    for _, row in df.iterrows():
        symbol = str(row["Symbol"]).upper().strip()
        company = str(row["Security"]).upper().strip()
        # On mappe toujours le ticker vers lui‑même
        symbol_map[symbol] = symbol
        # On supprime le texte entre parenthèses (« Alphabet Inc. (Class A) » → « Alphabet Inc. »)
        base = re.sub(r"\s*\(.*?\)", "", company).strip()
        # On génère plusieurs variantes : nom complet, nom sans parenthèses…
        candidates = {company, base}
        # On retire les suffixes communs
        simplified = suffix_re.sub("", base).strip()
        if simplified:
            candidates.add(simplified)
        # On enlève toute ponctuation et espaces superflus
        candidates = {re.sub(r"[^A-Z0-9 ]", "", c).strip() for c in candidates}
        # Chaque variante est associée au ticker
        for name in candidates:
            if name and name not in symbol_map:
                symbol_map[name] = symbol

    # Aliases manuels pour les marques populaires
    manual_aliases = {
        "GOOGLE": "GOOGL",
        "FACEBOOK": "META",
        "META": "META",
        "TESLA": "TSLA",
        "NETFLIX": "NFLX",
        "AMAZON": "AMZN",
        "APPLE": "AAPL",
        "MICROSOFT": "MSFT",
        "NVIDIA": "NVDA",
        "ALPHABET": "GOOGL",
        "GOOG": "GOOG",
        "GOOGL": "GOOGL",
        "META PLATFORMS": "META",
        "WALMART": "WMT",
        "UNITEDHEALTH": "UNH",
    }
    for alias, sym in manual_aliases.items():
        symbol_map[alias] = sym
    return symbol_map

def detect_symbol(text: str, symbol_map: dict) -> str | None:
    """
    Détecte le ticker le plus probable à partir du texte d’un article.
    La recherche s’effectue sur l’ensemble des alias connus via des
    correspondances mot‑entier. Les noms les plus longs sont testés en premier
    pour réduire les faux positifs (par exemple, éviter que le ticker « A »
    corresponde à n’importe quel mot). Les tickers d’un seul caractère sont
    uniquement repérés s’ils sont précédés d’un signe $ ou placés entre
    parenthèses.
    """
    if not text:
        return None
    text_upper = text.upper()
    
    # D'abord, chercher les noms longs (4+ caractères) - plus fiables
    for name in sorted(symbol_map.keys(), key=len, reverse=True):
        if len(name) >= 4:
            symbol = symbol_map[name]
            pattern = rf"\b{re.escape(name)}\b"
            if re.search(pattern, text_upper):
                return symbol
    
    # Ensuite, chercher les tickers de 2-3 caractères avec contexte
    for name in sorted(symbol_map.keys(), key=len, reverse=True):
        if 2 <= len(name) <= 3:
            symbol = symbol_map[name]
            # Chercher avec $ ou () ou après des mots clés financiers
            ticker_pattern = rf"(\$|NYSE:|NASDAQ:|TICKER:|STOCK:|SHARES OF|\()\s*{re.escape(name)}\s*(\)|$|\s)"
            if re.search(ticker_pattern, text_upper):
                return symbol
    
    # Enfin, tickers d'un seul caractère uniquement avec $ ou ()
    for name in sorted(symbol_map.keys(), key=len, reverse=True):
        if len(name) == 1:
            symbol = symbol_map[name]
            if re.search(rf"(\$|\()\s*{re.escape(name)}\s*(\)|\b)", text_upper):
                return symbol
    
    return None

def main():
    # Charger dynamiquement les symboles S&P500
    SYMBOLS = get_symbol_map()

    consumer = KafkaConsumer(
        KAFKA_CONFIG["topics"]["raw_news"],
        bootstrap_servers=KAFKA_CONFIG["bootstrap_servers"],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_CONFIG["bootstrap_servers"],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        **KAFKA_CONFIG["producer_config"]
    )

    model = SentimentModel()
    print("✅ Sentiment Analysis Consumer is running...")

    for msg in consumer:
        try:
            article = msg.value
            print(f"🔍 Received article: {article['title']}")

            cleaned_content = clean_text(article["content"])
            score = model.predict_sentiment(cleaned_content)

            # Détection automatique du symbole
            symbol = detect_symbol(article.get("content",""), SYMBOLS)

            if not symbol:
                print("⛔ Aucun symbole détecté dans l’article, article ignoré.")
                continue

            enriched_article = {
                **article,
                "sentiment_score": float(score),
                "symbol": symbol
            }

            producer.send(KAFKA_CONFIG["topics"]["news_sentiment"], value=enriched_article)
            print(f"📤 Sent enriched article with score {score} and symbol {symbol}")

            with open('/home/saadyaq/SE/Python/finsentbot/data/raw/news_sentiment.jsonl', "a") as f:
                f.write(json.dumps(enriched_article) + '\n')

        except Exception as e:
            print(f"⚠️ Error processing message: {e}")

if __name__ == "__main__":
    main()
