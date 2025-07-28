import json
from kafka import KafkaConsumer, KafkaProducer
import pandas as pd
from config.kafka_config import KAFKA_CONFIG

from nlp.preprocessing import clean_text
from nlp.sentiment_model import SentimentModel

def get_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    return df['Symbol'].tolist()

def detect_symbol(text, symbols_list):
    text = text.upper()
    for symbol in symbols_list:
        if symbol in text:
            return symbol
    return None

def main():
    # Charger dynamiquement les symboles S&P500
    SYMBOLS = get_sp500_symbols()

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
            symbol = detect_symbol(article["content"], SYMBOLS)

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
