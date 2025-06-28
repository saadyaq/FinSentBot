from kafka import KafkaProducer
import yfinance as yf 
import json
import time 
from datetime import datetime
from config.kafka_config import KAFKA_CONFIG

producer = KafkaProducer(
    bootstrap_servers=KAFKA_CONFIG["bootstrap_servers"],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    **KAFKA_CONFIG["producer_config"]
)

SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]

while True :

    try:
        print("🔍 Fetching stock prices for:", SYMBOLS)
        tickers = yf.Tickers(' '.join(SYMBOLS))

        for symbol in SYMBOLS:
            data = tickers.tickers[symbol].history(period='1d', interval='1m')

            if data.empty:
                print(f"⚠️ No data for {symbol}")
                continue

            current_price = data['Close'].iloc[-1]

            message = {
                "symbol": symbol,
                "price": round(current_price, 2),
                "timestamp": datetime.utcnow().isoformat()
            }

            producer.send(KAFKA_CONFIG["topics"]["stock_prices"], value=message)
            print(f"📤 Sent: {message}")

        print("✅ Sleeping 10 seconds...\n")
        time.sleep(10)

    except Exception as e :
        print(f"⚠️ Error: {e}")
        time.sleep(10)