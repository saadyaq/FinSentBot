from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Répertoires où se trouvent les données
BASE_DIR = Path(__file__).resolve().parent.parent
NEWS_PATH = BASE_DIR / "data" / "raw" / "news_sentiment.jsonl"
PRICES_PATH = BASE_DIR / "data" / "raw" / "stock_prices.jsonl"

# Fenêtre d'observation après la news
OBSERVATION_WINDOW_MINUTES = 3  # Ajusté selon les données disponibles
# Tolérance pour l'association prix/news
MERGE_TOLERANCE_MINUTES = 10

# Seuils pour les actions (en pourcentage) - ajustés pour 3 min
BUY_THRESHOLD = 0.002   # +0.2% ou plus
SELL_THRESHOLD = -0.002  # -0.2% ou moins
# Entre -0.2% et +0.2% = HOLD

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

        # Utiliser le prix déjà mergé dans news_df
        if pd.isna(row.get("price")):
            print(f"⛔ Aucun prix courant pour {symbol} à {timestamp}")
            continue

        p0 = row["price"]  # Prix au moment de la news (déjà mergé)

        # Prix après la fenêtre d'observation
        price_future = prices_df[
            (prices_df["symbol"] == symbol) &
            (prices_df["timestamp"] > timestamp + timedelta(minutes=OBSERVATION_WINDOW_MINUTES))
        ].sort_values("timestamp").head(1)

        if price_future.empty:
            print(f"⛔ Aucun prix futur pour {symbol} à {timestamp + timedelta(minutes=OBSERVATION_WINDOW_MINUTES)}")
            continue

        p1 = price_future.iloc[0]["price"]
        variation = (p1 - p0) / p0

        # Logique d'attribution simplifiée et équilibrée
        if variation >= BUY_THRESHOLD:
            action = "BUY"
        elif variation <= SELL_THRESHOLD:
            action = "SELL"
        else:
            # Variation faible: utiliser le sentiment
            if sentiment > 0.1:  # Sentiment positif
                action = "BUY"
            elif sentiment < -0.1:  # Sentiment négatif
                action = "SELL"
            else:
                action = "HOLD"  # Sentiment neutre

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
    new_dataset = generate_labels(news_df, prices_df)

    output_dir = BASE_DIR / "data" / "training_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "train.csv"
    
    # Charger le dataset existant s'il existe
    if output_file.exists():
        try:
            existing_dataset = pd.read_csv(output_file)
            print(f"📊 Dataset existant: {len(existing_dataset)} échantillons")
            
            # Éviter les doublons en utilisant symbol+text comme clé unique
            new_dataset['key'] = new_dataset['symbol'] + '|' + new_dataset['text'].str[:100]
            existing_dataset['key'] = existing_dataset['symbol'] + '|' + existing_dataset['text'].str[:100]
            
            # Filtrer les nouveaux échantillons uniquement
            mask = ~new_dataset['key'].isin(existing_dataset['key'])
            truly_new = new_dataset[mask].drop('key', axis=1)
            
            if len(truly_new) > 0:
                # Combiner les datasets
                combined_dataset = pd.concat([existing_dataset.drop('key', axis=1), truly_new], ignore_index=True)
                print(f"➕ {len(truly_new)} nouveaux échantillons ajoutés")
            else:
                combined_dataset = existing_dataset.drop('key', axis=1)
                print("ℹ️ Aucun nouvel échantillon à ajouter")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement du dataset existant: {e}")
            combined_dataset = new_dataset
    else:
        combined_dataset = new_dataset
        print("📝 Création d'un nouveau dataset")
    
    # Sauvegarder le dataset combiné
    combined_dataset.to_csv(output_file, index=False)
    
    # Statistiques finales
    action_counts = combined_dataset['action'].value_counts()
    print(f"✅ Dataset final: {len(combined_dataset)} échantillons")
    print(f"📈 Répartition: {dict(action_counts)}")
