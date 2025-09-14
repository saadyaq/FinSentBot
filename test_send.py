from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',  # ← à adapter si tu es DANS un container
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

message = {
    "title": "Test News",
    "text": "The market is reacting positively.",
    "timestamp": "2025-07-23T22:00:00Z"
}

future = producer.send("raw_news", value=message)
result = future.get(timeout=10)
producer.flush()

print("✅ Message sent to raw_news | topic:", result.topic, "| partition:", result.partition)


