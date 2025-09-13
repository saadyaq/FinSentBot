import time
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import json

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# Financial hashtags and keywords to search for
FINANCIAL_HASHTAGS = [
    "#stocks", "#trading", "#investing", "#finance", "#wallstreet", 
    "#stockmarket", "#NYSE", "#NASDAQ", "#earnings", "#dividend",
    "#options", "#futures", "#crypto", "#bitcoin", "#ethereum"
]

# Use alternative Twitter frontends for scraping
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.it", 
    "https://nitter.fdn.fr"
]

def fetch_twitter_posts():
    """
    Fetch Twitter posts related to financial topics using Nitter instances
    Returns list of (text, url) tuples
    """
    posts = []
    
    for instance in NITTER_INSTANCES:
        try:
            for hashtag in FINANCIAL_HASHTAGS[:3]:  # Limit to avoid rate limiting
                search_url = f"{instance}/search?f=tweets&q={hashtag}"
                
                try:
                    response = requests.get(search_url, headers=headers, timeout=10)
                    if response.status_code != 200:
                        continue
                        
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find tweet containers
                    tweets = soup.find_all('div', class_='timeline-item')
                    
                    for tweet in tweets[:10]:  # Limit per hashtag
                        try:
                            # Extract tweet text
                            tweet_text_elem = tweet.find('div', class_='tweet-content')
                            if not tweet_text_elem:
                                continue
                                
                            tweet_text = tweet_text_elem.get_text(strip=True)
                            
                            # Skip if too short
                            if len(tweet_text) < 50:
                                continue
                            
                            # Extract tweet URL
                            tweet_link = tweet.find('a', class_='tweet-link')
                            tweet_url = ""
                            if tweet_link:
                                tweet_url = instance + tweet_link.get('href', '')
                            
                            # Check if tweet contains stock symbols or financial terms
                            if contains_financial_content(tweet_text):
                                posts.append((tweet_text, tweet_url))
                                
                        except Exception as e:
                            print(f"Error processing tweet: {e}")
                            continue
                    
                    time.sleep(2)  # Rate limiting
                    
                except Exception as e:
                    print(f"Error fetching from {search_url}: {e}")
                    continue
            
            # If we got some posts from this instance, break
            if posts:
                break
                
        except Exception as e:
            print(f"Error with instance {instance}: {e}")
            continue
    
    # Remove duplicates
    unique_posts = list(dict.fromkeys(posts))
    print(f"[✓] {len(unique_posts)} unique Twitter posts extracted.")
    return unique_posts

def contains_financial_content(text):
    """Check if text contains financial terms or stock tickers"""
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Financial keywords
    financial_terms = [
        'stock', 'stocks', 'trading', 'buy', 'sell', 'earnings', 
        'dividend', 'market', 'bullish', 'bearish', 'calls', 'puts',
        'portfolio', 'investment', 'revenue', 'profit', 'loss'
    ]
    
    # Check for financial terms
    for term in financial_terms:
        if term in text_lower:
            return True
    
    # Check for stock ticker pattern ($SYMBOL)
    ticker_pattern = r'\$[A-Z]{1,5}\b'
    if re.search(ticker_pattern, text):
        return True
    
    return False

def extract_post_content(url):
    """
    Extract full content from a Twitter post URL
    For Twitter, the content is already in the post text
    """
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return ""
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to extract more detailed tweet content
        tweet_content = soup.find('div', class_='tweet-content')
        if tweet_content:
            return tweet_content.get_text(strip=True)
        
        return ""
        
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return ""

def main():
    """Test the Twitter scraper"""
    posts = fetch_twitter_posts()
    
    for text, url in posts[:5]:  # Show first 5
        print(f"\nText: {text[:100]}...")
        print(f"URL: {url}")
        print("-" * 50)

if __name__ == "__main__":
    main()
    print("[✓] Twitter scraping test completed.")