import requests
from bs4 import BeautifulSoup
import pandas as pd 
import sqlite3
import time
import os 
import re

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR=os.path.join(BASE_DIR,"data")
os.makedirs(DATA_DIR,exist_ok=True)

# Configuration
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
}

MARKETWATCH_URLS = [
    "https://www.marketwatch.com/latest-news",
    "https://www.marketwatch.com/markets",
    "https://www.marketwatch.com/economy-politics"
]

def fetch_marketwatch_article_links(max_articles=50):
    """Fetch MarketWatch article links from multiple sections"""

    all_links=[]
    for base_url in MARKETWATCH_URLS:
        try:
            print(f"Scraping {base_url}")
            response=requests.get(base_url,headers=headers,timeout=10)
            soup=BeautifulSoup(response.text,"html.parser")

            link_selectors=[
                'h3.article__headline a',
                'h2.article__headline a', 
                'a.link[href*="/story/"]',
                'a[href*="/articles/"]',
                '.article-wrap h3 a',
                '.headline a',
                'a.WSJTheme--headline-link'
            ]

            for selector in link_selectors:
                elements=soup.select(selector)
                for elem in elements:
                    href=elem.get("href","")
                    title=elem.get_text(strip=True)

                    if href and title and len(title)>25:
                        if href.startswith("/"):
                            href = f"https://www.marketwatch.com{href}"
                        elif not href.startswith("http"):
                            continue
                        if ("marketwatch.com" in href and 
                            ("/story/" in href or "/articles/" in href) and
                            title not in [link[0] for link in all_links]):
                            all_links.append((title, href))
            
            time.sleep(1)
        except Exception as e:
            print(f"Error scraping {base_url}: {e}")
            continue
    
    return all_links[:max_articles]

def extract_article_content(url):
    """Extract article content from MarketWatch article page"""

    try:
        response=requests.get(url,headers=headers,timeout=15)
        soup=BeautifulSoup(response.text,"html.parser")
        
        # Updated selectors based on current MarketWatch structure
        content_selectors = [
            'div.articleBody p',
            'div.entry-content p',
            'div.article-body p',
            'div.article__content p',
            'div.WSJTheme--article-body p',
            '.ArticleBody-articleBody p',
            '#article-body p'
        ]
        content=""
        for selector in content_selectors:
            elements=soup.select(selector)
            if elements:
                paragraphs=[]

                for p in elements:
                    text=p.get_text(strip=True)
                    if (text and 
                        "Subscribe to MarketWatch" not in text and
                        "Read the full story" not in text and
                        len(text) > 20):
                        paragraphs.append(text)
                
                content = " ".join(paragraphs)
                break
        
        # Fallback: Get all paragraphs with substantial content
        if not content:
            all_paragraphs = soup.find_all("p")
            paragraphs = []
            
            for p in all_paragraphs:
                text = p.get_text(strip=True)
                if (text and len(text) > 50 and 
                    "Subscribe" not in text and
                    "Read the full story" not in text and
                    "MarketWatch" not in text and
                    "Also read:" not in text and
                    "Don't miss:" not in text and
                    "© " not in text):
                    paragraphs.append(text)
            
            content = " ".join(paragraphs)
        
        # Additional fallback: Look for main content areas
        if not content:
            main_content = soup.find("main") or soup.find("article")
            if main_content:
                paragraphs = main_content.find_all("p")
                content = " ".join(p.get_text(strip=True) for p in paragraphs 
                                if p.get_text(strip=True) and len(p.get_text(strip=True)) > 20)
        
        # Clean up content
        content = re.sub(r'\s+', ' ', content)  # Multiple whitespace
        content = content.replace("Also read:", "")
        content = content.replace("Don't miss:", "")
        
        return content.strip()
        
    except Exception as e:
        print(f"[!] Error extracting content from {url}: {e}")
        return ""

def save_articles_to_db(df, db_path=os.path.join(DATA_DIR, "articles.db")):
    """Save articles to SQLite database"""
    conn = sqlite3.connect(db_path)
    df.to_sql("articles", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"[✓] {len(df)} MarketWatch articles saved to database")

def main():
    """Main scraping pipeline for MarketWatch"""
    articles = []
    links = fetch_marketwatch_article_links()
    print(f"Processing {len(links)} MarketWatch articles...")
    
    for title, url in links:
        print(f"Scraping: {title[:60]}...")
        content = extract_article_content(url)
        time.sleep(2)  # Respectful rate limiting
        
        if len(content.strip()) < 300:
            print(f"[!] Content too short for: {title[:40]}...")
            continue
            
        articles.append({
            "title": title,
            "content": content,
            "summary": None,
            "url": url,
            "source": "MarketWatch",
            "date": pd.Timestamp.now().strftime("%Y-%m-%d")
        })
    
    if articles:
        df = pd.DataFrame(articles)
        save_articles_to_db(df)
        print(f"[✅] MarketWatch scraping completed: {len(articles)} articles")
    else:
        print("[!] No valid MarketWatch articles found")

if __name__ == "__main__":
    main()