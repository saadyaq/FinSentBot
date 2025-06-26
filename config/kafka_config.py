KAFKA_CONFIG={
    "bootstrap_servers": "localhost:9092",
    "topics":{
        "stock_prices":"stock_prices",
        "raw_news":"raw_news",
        "news_sentiment":"news_sentiment",
        "trading_signals":"trading_signals"
    },
    "producer_config":{
        "acks":"all",
        "retries":3
    }
}