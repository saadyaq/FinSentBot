import time
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import json

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# Target subreddits for financial discussions
FINANCIAL_SUBREDDITS = [
    "stocks",
    "investing", 
    "SecurityAnalysis",
    "ValueInvesting",
    "financialindependence",
    "StockMarket",
    "options",
    "dividends"
]

# Reddit alternatives that don't require API
REDDIT_ALTERNATIVES = [
    "https://old.reddit.com",
    "https://www.reddit.com"
]

def fetch_reddit_posts():
    """
    Fetch Reddit posts from financial subreddits
    Returns list of (title, url, content) tuples
    """
    posts = []
    
    for subreddit in FINANCIAL_SUBREDDITS[:4]:  # Limit subreddits to avoid rate limiting
        for base_url in REDDIT_ALTERNATIVES:
            try:
                # Try both hot and new sorting
                for sort_type in ['hot', 'new']:
                    subreddit_url = f"{base_url}/r/{subreddit}/{sort_type}/.json"
                    
                    try:
                        response = requests.get(subreddit_url, headers=headers, timeout=15)
                        if response.status_code == 200:
                            data = response.json()
                            
                            if 'data' in data and 'children' in data['data']:
                                for post in data['data']['children'][:15]:  # Limit posts per subreddit
                                    try:
                                        post_data = post['data']
                                        
                                        title = post_data.get('title', '')
                                        selftext = post_data.get('selftext', '')
                                        permalink = post_data.get('permalink', '')
                                        
                                        # Skip if title too short or deleted/removed
                                        if len(title) < 20 or selftext in ['[deleted]', '[removed]']:
                                            continue
                                        
                                        # Combine title and content
                                        full_content = f"{title}. {selftext}"
                                        
                                        # Skip if content too short
                                        if len(full_content.strip()) < 100:
                                            continue
                                        
                                        # Check if contains financial content
                                        if contains_financial_content(full_content):
                                            full_url = f"{base_url}{permalink}"
                                            posts.append((title, full_url, full_content))
                                    
                                    except Exception as e:
                                        print(f"Error processing Reddit post: {e}")
                                        continue
                            
                            time.sleep(2)  # Rate limiting
                            break  # If successful, don't try other alternatives
                            
                    except requests.exceptions.JSONDecodeError:
                        # Try HTML scraping as fallback
                        html_url = f"{base_url}/r/{subreddit}/{sort_type}/"
                        posts.extend(scrape_reddit_html(html_url))
                        break
                        
                    except Exception as e:
                        print(f"Error fetching from {subreddit_url}: {e}")
                        continue
                
                if posts:
                    break  # If we got posts, move to next subreddit
                    
            except Exception as e:
                print(f"Error with subreddit {subreddit}: {e}")
                continue
    
    # Remove duplicates
    unique_posts = []
    seen_titles = set()
    for title, url, content in posts:
        if title not in seen_titles:
            unique_posts.append((title, url, content))
            seen_titles.add(title)
    
    print(f"[✓] {len(unique_posts)} unique Reddit posts extracted.")
    return unique_posts

def scrape_reddit_html(url):
    """
    Fallback HTML scraping for Reddit when JSON API fails
    """
    posts = []
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return posts
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find post containers (different selectors for old vs new Reddit)
        post_containers = soup.find_all('div', class_=['thing', 'Post'])
        
        for container in post_containers[:10]:  # Limit posts
            try:
                # Extract title
                title_elem = container.find('a', class_=['title', 'SQnoC3ObvgnGjWt90zD9Z'])
                if not title_elem:
                    continue
                    
                title = title_elem.get_text(strip=True)
                
                if len(title) < 20:
                    continue
                
                # Extract URL
                post_url = title_elem.get('href', '')
                if post_url.startswith('/'):
                    post_url = 'https://old.reddit.com' + post_url
                
                # For HTML scraping, we'll use the title as content
                # Full content extraction would require another request
                content = title
                
                if contains_financial_content(content):
                    posts.append((title, post_url, content))
                    
            except Exception as e:
                print(f"Error processing HTML post: {e}")
                continue
        
        time.sleep(3)  # Extra delay for HTML scraping
        
    except Exception as e:
        print(f"Error scraping HTML from {url}: {e}")
    
    return posts

def contains_financial_content(text):
    """Check if text contains financial terms or stock discussions"""
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Financial keywords
    financial_terms = [
        'stock', 'stocks', 'share', 'shares', 'trading', 'buy', 'sell', 
        'earnings', 'dividend', 'market', 'bullish', 'bearish', 'portfolio',
        'investment', 'investing', 'revenue', 'profit', 'loss', 'dd', 
        'due diligence', 'analysis', 'valuation', 'pe ratio', 'eps',
        'options', 'calls', 'puts', 'strike', 'expiry', 'volatility'
    ]
    
    # Check for financial terms
    for term in financial_terms:
        if term in text_lower:
            return True
    
    # Check for stock ticker patterns
    ticker_patterns = [
        r'\$[A-Z]{1,5}\b',  # $AAPL format
        r'\b[A-Z]{1,5}\b',  # AAPL format (more risky, many false positives)
        r'ticker:?\s*[A-Z]{1,5}\b'  # ticker: AAPL format
    ]
    
    for pattern in ticker_patterns[:1]:  # Only use first pattern to reduce false positives
        if re.search(pattern, text):
            return True
    
    return False

def extract_post_content(url):
    """
    Extract full content from a Reddit post URL
    """
    try:
        # Convert to JSON API URL if possible
        if '/comments/' in url:
            json_url = url + '.json'
            
            response = requests.get(json_url, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                
                if len(data) > 0 and 'data' in data[0] and 'children' in data[0]['data']:
                    post_data = data[0]['data']['children'][0]['data']
                    
                    title = post_data.get('title', '')
                    selftext = post_data.get('selftext', '')
                    
                    return f"{title}. {selftext}"
        
        # Fallback to HTML scraping
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return ""
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find post content
        content_elem = soup.find('div', class_=['usertext-body', 'RichTextJSON-root'])
        if content_elem:
            return content_elem.get_text(strip=True)
        
        return ""
        
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return ""

def main():
    """Test the Reddit scraper"""
    posts = fetch_reddit_posts()
    
    for title, url, content in posts[:3]:  # Show first 3
        print(f"\nTitle: {title}")
        print(f"Content: {content[:200]}...")
        print(f"URL: {url}")
        print("-" * 80)

if __name__ == "__main__":
    main()
    print("[✓] Reddit scraping test completed.")