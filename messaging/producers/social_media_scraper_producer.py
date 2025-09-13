import json 
import time 
import re
from datetime import datetime
from kafka import KafkaProducer
from config.kafka_config import KAFKA_CONFIG

# === Import social media scrapers ===
from messaging.producers.scrapers.src.scraper_twitter import fetch_twitter_posts, extract_post_content as extract_twitter
from messaging.producers.scrapers.src.scraper_reddit import fetch_reddit_posts, extract_post_content as extract_reddit
from messaging.producers.scrapers.src.scraper_stocktwits import fetch_stocktwits_posts, extract_post_content as extract_stocktwits

# === S&P 500 Company Keywords (same as news scraper) ===
SP500_KEYWORDS = {
    # Major tech companies
    'AAPL': ['apple', 'iphone', 'ipad', 'mac', 'ios', 'app store'],
    'MSFT': ['microsoft', 'windows', 'azure', 'office', 'teams'],
    'GOOGL': ['google', 'alphabet', 'youtube', 'android', 'gmail', 'search'],
    'AMZN': ['amazon', 'aws', 'alexa', 'prime', 'kindle'],
    'TSLA': ['tesla', 'elon musk', 'electric vehicle', 'ev', 'model'],
    'META': ['meta', 'facebook', 'instagram', 'whatsapp', 'metaverse'],
    'NVDA': ['nvidia', 'gpu', 'ai chip', 'graphics card'],
    'NFLX': ['netflix', 'streaming', 'original series'],
    
    # Finance
    'JPM': ['jpmorgan', 'jp morgan', 'chase bank'],
    'BAC': ['bank of america', 'bofa'],
    'WFC': ['wells fargo'],
    'GS': ['goldman sachs'],
    'MS': ['morgan stanley'],
    
    # Healthcare
    'JNJ': ['johnson & johnson', 'johnson and johnson', 'j&j'],
    'UNH': ['unitedhealth', 'united health'],
    'PFE': ['pfizer'],
    'ABBV': ['abbvie'],
    
    # Consumer
    'WMT': ['walmart', 'wal-mart'],
    'HD': ['home depot'],
    'MCD': ['mcdonald', "mcdonald's"],
    'KO': ['coca-cola', 'coca cola', 'coke'],
    'PEP': ['pepsi', 'pepsico'],
    'NKE': ['nike'],
    
    # Industrial/Energy
    'XOM': ['exxon', 'exxonmobil'],
    'CVX': ['chevron'],
    'BA': ['boeing'],
    'CAT': ['caterpillar'],
    'GE': ['general electric'],
    
    # ETFs and Indices
    'SPY': ['spy', 's&p 500', 'sp500', 'spdr'],
    'QQQ': ['qqq', 'nasdaq', 'tech etf'],
    'IWM': ['iwm', 'russell 2000', 'small cap'],
}

def contains_sp500_mention(text):
    """Check if text contains mentions of S&P 500 companies or popular tickers"""
    if not text:
        return False, []
    
    text_lower = text.lower()
    mentioned_symbols = []
    
    # Check keyword matches
    for symbol, keywords in SP500_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                mentioned_symbols.append(symbol)
                break
    
    # Check for direct ticker mentions (both $TICKER and TICKER formats)
    ticker_patterns = [
        r'\$([A-Z]{1,5})\b',  # $AAPL format
        r'\b([A-Z]{2,5})\b'   # AAPL format (more restrictive to avoid false positives)
    ]
    
    for pattern in ticker_patterns:
        tickers = re.findall(pattern, text)
        for ticker in tickers:
            if ticker in SP500_KEYWORDS:
                mentioned_symbols.append(ticker)
    
    return len(mentioned_symbols) > 0, list(set(mentioned_symbols))

# === Kafka producer configuration ===
producer = KafkaProducer(
    bootstrap_servers=KAFKA_CONFIG["bootstrap_servers"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    **KAFKA_CONFIG["producer_config"]
)

def scrape_twitter_and_send():
    """Scrape Twitter posts and send to Kafka"""
    print("🐦 Scraping Twitter...")
    
    try:
        posts = fetch_twitter_posts()
        print(f"[✓] {len(posts)} Twitter posts found")
        
        sent_count = 0
        for text, url in posts:
            # Check for S&P 500 company mentions
            has_sp500, mentioned_symbols = contains_sp500_mention(text)
            
            # Accept posts with financial relevance even without specific S&P 500 mentions
            if not has_sp500:
                # For social media, we're more lenient - check for general financial terms
                financial_terms = ['stock', 'trading', 'market', 'buy', 'sell', 'bullish', 'bearish']
                if not any(term in text.lower() for term in financial_terms):
                    continue
                mentioned_symbols = ['GENERAL']  # Mark as general financial content
            
            message = {
                "source": "Twitter",
                "title": text[:100] + "..." if len(text) > 100 else text,  # Use first part as title
                "content": text,
                "url": url,
                "timestamp": datetime.utcnow().isoformat(),
                "mentioned_symbols": mentioned_symbols,
                "platform": "social_media"
            }
            
            producer.send(KAFKA_CONFIG["topics"]["raw_news"], value=message)
            print(f"📤 Sent Twitter post [{','.join(mentioned_symbols)}]: {text[:60]}...")
            sent_count += 1
        
        print(f"[✓] {sent_count} Twitter posts sent to Kafka")
        
    except Exception as e:
        print(f"⚠️ Error scraping Twitter: {e}")

def scrape_reddit_and_send():
    """Scrape Reddit posts and send to Kafka"""
    print("🔴 Scraping Reddit...")
    
    try:
        posts = fetch_reddit_posts()
        print(f"[✓] {len(posts)} Reddit posts found")
        
        sent_count = 0
        for title, url, content in posts:
            # Check for S&P 500 company mentions
            full_text = f"{title} {content}"
            has_sp500, mentioned_symbols = contains_sp500_mention(full_text)
            
            if not has_sp500:
                # For Reddit, check if it's from financial subreddits (implied financial relevance)
                if any(sub in url for sub in ['r/stocks', 'r/investing', 'r/SecurityAnalysis']):
                    mentioned_symbols = ['GENERAL']
                else:
                    continue
            
            message = {
                "source": "Reddit",
                "title": title,
                "content": content,
                "url": url,
                "timestamp": datetime.utcnow().isoformat(),
                "mentioned_symbols": mentioned_symbols,
                "platform": "social_media"
            }
            
            producer.send(KAFKA_CONFIG["topics"]["raw_news"], value=message)
            print(f"📤 Sent Reddit post [{','.join(mentioned_symbols)}]: {title[:60]}...")
            sent_count += 1
        
        print(f"[✓] {sent_count} Reddit posts sent to Kafka")
        
    except Exception as e:
        print(f"⚠️ Error scraping Reddit: {e}")

def scrape_stocktwits_and_send():
    """Scrape StockTwits posts and send to Kafka"""
    print("📈 Scraping StockTwits...")
    
    try:
        posts = fetch_stocktwits_posts()
        print(f"[✓] {len(posts)} StockTwits posts found")
        
        sent_count = 0
        for message_text, symbol, url in posts:
            # StockTwits posts are inherently tied to specific symbols
            mentioned_symbols = [symbol] if symbol else ['GENERAL']
            
            message = {
                "source": "StockTwits",
                "title": f"${symbol}: {message_text[:80]}..." if len(message_text) > 80 else f"${symbol}: {message_text}",
                "content": message_text,
                "url": url,
                "timestamp": datetime.utcnow().isoformat(),
                "mentioned_symbols": mentioned_symbols,
                "platform": "social_media"
            }
            
            producer.send(KAFKA_CONFIG["topics"]["raw_news"], value=message)
            print(f"📤 Sent StockTwits post [${symbol}]: {message_text[:50]}...")
            sent_count += 1
        
        print(f"[✓] {sent_count} StockTwits posts sent to Kafka")
        
    except Exception as e:
        print(f"⚠️ Error scraping StockTwits: {e}")

def main():
    """Main loop for social media scraping"""
    
    while True:
        print("\n=== 📱 Social Media Scraping Round Started ===\n")
        
        # Scrape each platform
        scrape_twitter_and_send()
        time.sleep(30)  # Brief pause between platforms
        
        scrape_reddit_and_send()
        time.sleep(30)
        
        scrape_stocktwits_and_send()
        
        print("\n✅ Social media scraping round complete. Sleeping for 30 minutes...\n")
        time.sleep(1800)  # 30 minutes between rounds (social media updates frequently)

if __name__ == "__main__":
    main()