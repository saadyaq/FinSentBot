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
    news_df = pd.read_json(NEWS_PATH, lines=True)
    prices_df = pd.read_json(PRICES_PATH, lines=True)

    # Convertir en datetime
    news_df["timestamp"] = pd.to_datetime(news_df["timestamp"]).dt.tz_localize(None)
    prices_df["timestamp"] = pd.to_datetime(prices_df["timestamp"]).dt.tz_localize(None)

    # Supprimer les news sans symbole pour éviter les associations aléatoires
    news_df = news_df.dropna(subset=["symbol"])


    # Trier pour merge_asof (timestamp puis symbole)
    news_df = news_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    prices_df = prices_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


    # Associer symbol et prix au moment de la news (merge asof)
    enriched_news = pd.merge_asof(
        news_df,
        prices_df[["timestamp", "symbol", "price"]],
        on="timestamp",
        by="symbol",
        direction="backward",
        tolerance=pd.Timedelta(minutes=MERGE_TOLERANCE_MINUTES)  # marge de tolérance temporelle
    )

    if "symbol" not in enriched_news.columns or "price" not in enriched_news.columns:
        raise ValueError("⛔ Les colonnes 'symbol' ou 'price' sont absentes du fichier news.")

    # Supprimer les news sans correspondance de prix
    news_df = enriched_news.dropna(subset=["symbol", "price"])
    return news_df, prices_df

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
