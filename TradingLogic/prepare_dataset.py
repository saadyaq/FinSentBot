from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Répertoires où se trouvent les données
BASE_DIR = Path(__file__).resolve().parent.parent
NEWS_PATH = BASE_DIR / "data" / "raw" / "news_sentiment.jsonl"
PRICES_PATH = BASE_DIR / "data" / "raw" / "stock_prices.jsonl"

# Fenêtre d'observation après la news
OBSERVATION_WINDOW_MINUTES = 2
# Tolérance pour l'association prix/news
MERGE_TOLERANCE_MINUTES = 10

def load_data():
    news_df   = pd.read_json(NEWS_PATH,   lines=True)
    prices_df = pd.read_json(PRICES_PATH, lines=True)

    # Timestamps normalisés (naïfs)
    news_df["timestamp"]   = pd.to_datetime(news_df["timestamp"], errors="coerce").dt.tz_localize(None)
    prices_df["timestamp"] = pd.to_datetime(prices_df["timestamp"], errors="coerce").dt.tz_localize(None)

    # DROP AVANT cast -> sinon NaN devient "nan"
    news_df   = news_df.dropna(subset=["symbol", "timestamp"])
    prices_df = prices_df.dropna(subset=["symbol", "timestamp", "price"])

    # Normalisation symboles
    news_df["symbol"]   = news_df["symbol"].astype(str).str.upper().str.strip()
    prices_df["symbol"] = prices_df["symbol"].astype(str).str.upper().str.strip()

    # Retirer les tokens invalides
    invalid = {"", "NAN", "NONE", "NULL"}
    news_df = news_df[~news_df["symbol"].isin(invalid)].copy()

    # On ne garde que les symboles vus dans les news
    symbols = sorted(news_df["symbol"].unique().tolist())
    prices_df = prices_df[prices_df["symbol"].isin(symbols)].copy()

    # Merge robuste: par symbole, groupe par groupe
    merged_parts, missing = [], []
    for sym in symbols:
        g_news   = news_df[news_df["symbol"] == sym].sort_values("timestamp", kind="mergesort")
        g_prices = prices_df[prices_df["symbol"] == sym].sort_values("timestamp", kind="mergesort")
        if g_prices.empty:
            missing.append(sym); continue
        m = pd.merge_asof(
            g_news,
            g_prices[["timestamp", "price"]],
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta("60min"),
            allow_exact_matches=True,
        )
        merged_parts.append(m)

    enriched_news = pd.concat(merged_parts, ignore_index=True) if merged_parts else news_df.iloc[0:0].copy()
    before = len(enriched_news)
    enriched_news = enriched_news.dropna(subset=["price"]).reset_index(drop=True)
    after = len(enriched_news)

    if missing:
        print("⚠️ Symboles présents dans les news mais absents des prix:", sorted(missing)[:20],
              f"(+{max(0, len(missing)-20)} autres)" if len(missing) > 20 else "")
    print(f"Après merge_asof par symbole: {after}/{before} news gardées avec prix")

    return enriched_news, prices_df



def generate_labels(news_df, prices_df):
    data = []
    for _, row in news_df.iterrows():
        timestamp = row["timestamp"]
        symbol = row["symbol"]
        sentiment = row.get("sentiment_score", 0)
        text = row["content"]

        # Prix courant au moment de l'info
        price_now = prices_df[
            (prices_df["symbol"] == symbol) &
            (prices_df["timestamp"] <= timestamp)
        ].sort_values("timestamp").tail(1)

        # Prix après 10 minutes
        price_future = prices_df[
            (prices_df["symbol"] == symbol) &
            (prices_df["timestamp"] > timestamp + timedelta(minutes=OBSERVATION_WINDOW_MINUTES))
        ].sort_values("timestamp").head(1)

        if price_now.empty :
            print(f"⛔ Aucun prix courant pour {symbol} à {timestamp}")
            continue
        if price_future.empty:
            print(f"⛔ Aucun prix futur pour {symbol} à {timestamp + timedelta(minutes=OBSERVATION_WINDOW_MINUTES)}")
            continue

        p0 = price_now.iloc[0]["price"]
        p1 = price_future.iloc[0]["price"]
        variation = (p1 - p0) / p0

        if variation > 0 and sentiment > 0:
            action = "BUY"
        elif variation < 0 and sentiment < 0:
            action = "SELL"
        else:
            action = "HOLD"

        data.append({
            "symbol": symbol,
            "text": text,
            "sentiment_score": sentiment,
            "price_now": p0,
            "price_future": p1,
            "variation": variation,
            "action": action
        })
        print(f"✅ Sample ajouté pour {symbol}, variation={variation:.4f}, action={action}")

    return pd.DataFrame(data)

if __name__ == "__main__":
    news_df, prices_df = load_data()
    train_dataset = generate_labels(news_df, prices_df)

    output_dir = BASE_DIR / "data" / "training_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.to_csv(output_dir / "train.csv", index=False)
    print("✅ Dataset generated with", len(train_dataset), "samples.")
