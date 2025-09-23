import time
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import json
from urllib.parse import quote_plus

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
    "https://nitter.privacydev.net",
    "https://nitter.1d4.us",
    "https://nitter.kavin.rocks"
]

def test_nitter_instance(instance):
    """Test if a Nitter instance is accessible"""
    try:
        response = requests.get(f"{instance}/about", headers=headers, timeout=5)
        return response.status_code == 200
    except:
        return False

def fetch_twitter_posts():
    """
    Fetch Twitter posts related to financial topics using Nitter instances
    Returns list of (text, url) tuples
    """
    posts = []
    working_instances = []
    
    # Test instances first
    print("Testing Nitter instances...")
    for instance in NITTER_INSTANCES:
        if test_nitter_instance(instance):
            working_instances.append(instance)
            print(f"[✓] {instance} is accessible")
        else:
            print(f"[✗] {instance} is not accessible")
    
    if not working_instances:
        print("[!] No working Nitter instances found. Twitter scraping disabled.")
        return []
    
    for instance in working_instances[:2]:  # Use max 2 working instances
        try:
            for hashtag in FINANCIAL_HASHTAGS[:3]:  # Limit to avoid rate limiting
                encoded_query = quote_plus(hashtag)  # Nitter expects encoded hash symbol
                search_url = f"{instance}/search?f=tweets&q={encoded_query}"
                
                try:
                    response = requests.get(search_url, headers=headers, timeout=15)
                    if response.status_code != 200:
                        print(f"[!] HTTP {response.status_code} for {search_url}")
                        continue
                        
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find tweet containers - try multiple selectors
                    tweets = (soup.find_all('div', class_='timeline-item') or 
                             soup.find_all('div', class_='tweet') or
                             soup.find_all('article'))
                    
                    for tweet in tweets[:10]:  # Limit per hashtag
                        try:
                            # Try multiple selectors for tweet content
                            tweet_text_elem = (tweet.find('div', class_='tweet-content') or
                                             tweet.find('div', class_='tweet-text') or
                                             tweet.find('p'))
                            
                            if not tweet_text_elem:
                                continue
                                
                            tweet_text = tweet_text_elem.get_text(strip=True)
                            
                            # Skip if too short
                            if len(tweet_text) < 30:
                                continue
                            
                            # Extract tweet URL
                            tweet_link = tweet.find('a', class_='tweet-link') or tweet.find('a')
                            tweet_url = ""
                            if tweet_link and tweet_link.get('href'):
                                href = tweet_link.get('href')
                                if href.startswith('/'):
                                    tweet_url = instance + href
                                else:
                                    tweet_url = href
                            
                            # Check if tweet contains stock symbols or financial terms
                            if contains_financial_content(tweet_text):
                                posts.append((tweet_text, tweet_url))
                                
                        except Exception as e:
                            print(f"Error processing tweet: {e}")
                            continue
                    
                    print(f"[+] Found {len(posts)} posts from {hashtag} on {instance}")
                    time.sleep(3)  # Rate limiting
                    
                except Exception as e:
                    print(f"Error fetching from {search_url}: {e}")
                    continue
            
            # If we got some posts from this instance, continue to next
            if len(posts) >= 10:  # Stop if we have enough posts
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
