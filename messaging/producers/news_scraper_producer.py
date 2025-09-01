import json 
import time 
import re
from datetime import datetime
from kafka import KafkaProducer
from config.kafka_config import KAFKA_CONFIG

# === Import of existing scrapers ===

from messaging.producers.scrapers.src.scraper_cnbc import fetch_cnbc_article_links , extract_article_content as extract_cnbc
from messaging.producers.scrapers.src.scraper_coindesk import fetch_coindesk_links, extract_article_content as extract_coindesk
from messaging.producers.scrapers.src.scraper_ft import fetch_ft_article_links , extract_article_content as extract_ft
from messaging.producers.scrapers.src.scraper_tc import fetch_tc_article_links, extract_article_content as extract_tc

# === S&P 500 Company Keywords ===
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
    'GE': ['general electric']
}

def contains_sp500_mention(text):
    """Check if text contains mentions of S&P 500 companies"""
    if not text:
        return False, []
    
    text_lower = text.lower()
    mentioned_symbols = []
    
    for symbol, keywords in SP500_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                mentioned_symbols.append(symbol)
                break
    
    # Also check for direct ticker mentions
    ticker_pattern = r'\b([A-Z]{1,5})\b'
    tickers = re.findall(ticker_pattern, text)
    for ticker in tickers:
        if ticker in SP500_KEYWORDS:
            mentioned_symbols.append(ticker)
    
    return len(mentioned_symbols) > 0, list(set(mentioned_symbols))

# === Config Kafka producer ===

producer = KafkaProducer(
    bootstrap_servers=KAFKA_CONFIG["bootstrap_servers"],
    value_serializer= lambda v: json.dumps(v).encode("utf-8"),
    **KAFKA_CONFIG["producer_config"]
)

def scrape_and_send(source, fetch_links, extract_func):

    """
    - Récupère les leins d'articles depuis un scraper
    - Extrait le contenu de chaque article
    - Envoie chaque message dans kafka 
    """

    print(f"🔍 Scraping {source}...")

    try :

        links = fetch_links()
        print(f"[✓] {len(links)} links trouvés pour {source}")

        for title , url in links :
            content = extract_func(url)
            time.sleep(1)

            if len(content.strip()) <300:
                continue 
            
            # Check for S&P 500 company mentions
            full_text = f"{title} {content}"
            has_sp500, mentioned_symbols = contains_sp500_mention(full_text)
            
            # Only send articles that mention S&P 500 companies
            if not has_sp500:
                print(f"⏭️  Skipped (no S&P 500 mention): {title[:60]}...")
                continue
            
            message = {
                "source": source,
                "title" : title,
                "content" : content,
                "url" : url, 
                "timestamp" : datetime.utcnow().isoformat(),
                "mentioned_symbols": mentioned_symbols  # Add detected symbols
            }

            producer.send(KAFKA_CONFIG["topics"]["raw_news"], value = message)
            print(f"📤 Sent to Kafka [{','.join(mentioned_symbols)}]: {title}")

    except Exception as e:
        print(f"⚠️ Error scraping {source}: {e}")

def main():

    while True:

        print("\n=== 🗞️ New Scraping Round Started ===\n")
        scrape_and_send("CNBC", fetch_cnbc_article_links, extract_cnbc)
        scrape_and_send("CoinDesk", fetch_coindesk_links, extract_coindesk)
        scrape_and_send("Financial Times", fetch_ft_article_links, extract_ft)
        scrape_and_send("TechCrunch", fetch_tc_article_links, extract_tc)
        
        print("\n✅ Scraping round complete. Sleeping for 20 minutes...\n")
        time.sleep(1200)  # toutes les 30 minutes

if __name__ == "__main__":
    main()