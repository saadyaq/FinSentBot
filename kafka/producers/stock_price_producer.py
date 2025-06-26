from kafka import KafkaProducer
import yfinance as yf 
import json
import time 
from datetime import datetime

producer= KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer= lambda v:json.dumps(v).encode('utf-8')
)

symbol= 'AAPL'

while True :

    try:
        stock= yf.Ticker(symbol)
        data= stock.history(period='1d', interval='1m')
        current_price=data['Close'].iloc[-1]

        message= {
            'symbol' :symbol,
            'price' : round(current_price,2),
            'timestamp': datetime.utcnow().isoformat()
        }

        producer.send("stock_prices", value=message)
        print(f"📤 Sent: {message}")

        time.sleep(10)

    except Exception as e :
        print(f"⚠️ Error: {e}")
        time.sleep(10)