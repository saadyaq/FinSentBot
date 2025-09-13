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
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://stocktwits.com/"
}

# Popular stock symbols to scrape from StockTwits
POPULAR_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
    "AMD", "INTC", "CRM", "ORCL", "ADBE", "PYPL", "SNOW", "PLTR",
    "JPM", "BAC", "GS", "WFC", "MS", "C",
    "SPY", "QQQ", "IWM", "VTI", "VOO"
]

def fetch_stocktwits_posts():
    """
    Fetch StockTwits posts for popular stock symbols
    Returns list of (message, symbol, url) tuples
    """
    posts = []
    
    for symbol in POPULAR_SYMBOLS[:10]:  # Limit symbols to avoid rate limiting
        try:
            # StockTwits symbol page URL
            symbol_url = f"https://stocktwits.com/symbol/{symbol}"
            
            response = requests.get(symbol_url, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"Failed to fetch {symbol}: {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # StockTwits uses React, so we need to find the data in script tags
            # Look for messages in JSON format within script tags
            script_tags = soup.find_all('script')
            
            for script in script_tags:
                if script.string and 'messages' in script.string:
                    try:
                        # Extract JSON data from script
                        script_content = script.string
                        
                        # Look for message patterns
                        message_patterns = re.finditer(r'"body":"([^"]+)"', script_content)
                        
                        for match in message_patterns:
                            message_text = match.group(1)
                            
                            # Decode escaped characters
                            message_text = message_text.replace('\\n', ' ').replace('\\/', '/').replace('\\"', '"')
                            
                            # Skip if too short
                            if len(message_text.strip()) < 30:
                                continue
                            
                            # Check if contains meaningful financial content
                            if contains_financial_content(message_text, symbol):
                                post_url = f"{symbol_url}#message"
                                posts.append((message_text, symbol, post_url))
                            
                            # Limit messages per symbol
                            if len([p for p in posts if p[1] == symbol]) >= 10:
                                break
                                
                    except Exception as e:
                        print(f"Error parsing script for {symbol}: {e}")
                        continue
            
            # Alternative: Try to scrape visible messages if JSON parsing fails
            if not any(p[1] == symbol for p in posts):
                posts.extend(scrape_visible_messages(soup, symbol, symbol_url))
            
            time.sleep(3)  # Rate limiting - StockTwits is strict
            
        except Exception as e:
            print(f"Error fetching StockTwits for {symbol}: {e}")
            continue
    
    # Remove duplicates
    unique_posts = []
    seen_messages = set()
    for message, symbol, url in posts:
        message_key = f"{symbol}_{message[:50]}"
        if message_key not in seen_messages:
            unique_posts.append((message, symbol, url))
            seen_messages.add(message_key)
    
    print(f"[✓] {len(unique_posts)} unique StockTwits posts extracted.")
    return unique_posts

def scrape_visible_messages(soup, symbol, base_url):
    """
    Fallback method to scrape visible messages from the page
    """
    posts = []
    
    try:
        # Look for message containers with various possible class names
        message_selectors = [
            'div[class*="message"]',
            'div[class*="post"]',
            'div[class*="stream"]',
            'article',
            'li[class*="StreamMessage"]'
        ]
        
        for selector in message_selectors:
            messages = soup.select(selector)
            
            if messages:
                for msg_elem in messages[:5]:  # Limit per selector
                    try:
                        message_text = msg_elem.get_text(strip=True)
                        
                        # Skip if too short or empty
                        if len(message_text.strip()) < 30:
                            continue
                        
                        # Clean up the message
                        message_text = re.sub(r'\s+', ' ', message_text)
                        
                        if contains_financial_content(message_text, symbol):
                            posts.append((message_text, symbol, base_url))
                            
                    except Exception as e:
                        continue
                
                # If we found messages with this selector, break
                if posts:
                    break
    
    except Exception as e:
        print(f"Error scraping visible messages for {symbol}: {e}")
    
    return posts

def contains_financial_content(text, symbol=None):
    """
    Check if text contains financial content relevant to the symbol
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Financial sentiment keywords
    financial_terms = [
        'bull', 'bullish', 'bear', 'bearish', 'buy', 'sell', 'hold',
        'long', 'short', 'calls', 'puts', 'strike', 'target', 'pt',
        'support', 'resistance', 'breakout', 'breakdown', 'rally',
        'dip', 'moon', 'rocket', 'diamond hands', 'hodl', 'yolo',
        'earnings', 'er', 'guidance', 'revenue', 'eps', 'beat', 'miss',
        'upgrade', 'downgrade', 'analyst', 'price target', 'rating'
    ]
    
    # Check for financial terms
    term_count = sum(1 for term in financial_terms if term in text_lower)
    
    # Must have at least 1 financial term
    if term_count == 0:
        return False
    
    # If symbol is provided, it should be mentioned or implied
    if symbol and symbol.lower() not in text_lower:
        # For StockTwits, the symbol is implied by the page context
        # So we accept posts even if symbol isn't explicitly mentioned
        pass
    
    # Check for stock ticker patterns
    ticker_patterns = [
        r'\$[A-Z]{1,5}\b',  # $AAPL format
        r'\b[A-Z]{2,5}\b'   # AAPL format (careful with false positives)
    ]
    
    has_ticker = any(re.search(pattern, text) for pattern in ticker_patterns)
    
    # Accept if has financial terms and either mentions a ticker or is symbol-specific
    return term_count >= 1 and (has_ticker or symbol)

def extract_post_content(url):
    """
    Extract content from StockTwits post URL
    For StockTwits, content is already captured in the message
    """
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return ""
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find specific message content
        # This is complex due to React-based structure
        content_selectors = [
            'div[class*="body"]',
            'div[class*="content"]', 
            'div[class*="message"]'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem.get_text(strip=True)
                if len(content) > 20:
                    return content
        
        return ""
        
    except Exception as e:
        print(f"Error extracting StockTwits content from {url}: {e}")
        return ""

def main():
    """Test the StockTwits scraper"""
    posts = fetch_stocktwits_posts()
    
    for message, symbol, url in posts[:5]:  # Show first 5
        print(f"\nSymbol: ${symbol}")
        print(f"Message: {message[:150]}...")
        print(f"URL: {url}")
        print("-" * 70)

if __name__ == "__main__":
    main()
    print("[✓] StockTwits scraping test completed.")