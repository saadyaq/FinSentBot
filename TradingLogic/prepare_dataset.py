import pandas as pd
from datetime import datetime, timedelta
import os

# Répertoires où se trouvent les données 
NEWS_PATH = "/home/saadyaq/SE/Python/finsentbot/data/raw/news_sentiment.jsonl"
PRICES_PATH = "/home/saadyaq/SE/Python/finsentbot/data/raw/stock_prices.jsonl"

# Fenêtre d'observation après la news
OBSERVATION_WINDOW_MINUTES = 2
THRESHOLD = 0.005

def load_data():
    news_df = pd.read_json(NEWS_PATH, lines=True)
    prices_df = pd.read_json(PRICES_PATH, lines=True)
    print(prices_df[prices_df["symbol"] == "ADP"]["timestamp"].sort_values().tail(10))
    print(news_df["timestamp"].max())
    print(prices_df["timestamp"].max())
    

    # Convertir en datetime
    news_df["timestamp"] = pd.to_datetime(news_df["timestamp"]).dt.tz_localize(None)
    prices_df["timestamp"] = pd.to_datetime(prices_df["timestamp"]).dt.tz_localize(None)
    # Trier pour merge_asof
    news_df = news_df.sort_values("timestamp")
    prices_df = prices_df.sort_values("timestamp")
    print("Avant Merge:", len(news_df), len(prices_df))
    # Associer symbol et prix au moment de la news (merge asof)
    enriched_news = pd.merge_asof(
        news_df,
        prices_df[["timestamp", "symbol", "price"]],
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("10min")  # ← marge de tolérance temporelle
    )
    print("Colonnes présentes :", enriched_news.columns)
    # Choisir l'une des deux colonnes symbol (selon celle qui est correcte)
    enriched_news["symbol"] = enriched_news["symbol_x"].combine_first(enriched_news["symbol_y"])

    if "symbol" not in enriched_news.columns or "price" not in enriched_news.columns:
        raise ValueError("⛔ Les colonnes 'symbol' ou 'price' sont absentes du fichier news.")
    
    news_df = enriched_news.dropna(subset=["symbol", "price"])
    # Supprimer les news sans correspondance de prix
    news_df = enriched_news.dropna(subset=["symbol", "price"])
    print("apres merge", len(news_df),len(prices_df))
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

        if variation > THRESHOLD:
            action = "BUY"
        elif variation < -THRESHOLD:
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

    os.makedirs("home/saadyaq/SE/finsentbot/data/training_datasets", exist_ok=True)
    train_dataset.to_csv("home/saadyaq/SE/finsentbot/data/training_datasets/train.csv", index=False)
    print("✅ Dataset generated with", len(train_dataset), "samples.")
