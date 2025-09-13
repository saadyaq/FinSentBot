import requests 
from bs4 import BeautifulSoup
import pandas as pd 
import sqlite3
import time
import json 
import os 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR=os.path.join(BASE_DIR,'data')
os.makedirs(DATA_DIR,exist_ok=True)

# Configuration
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://seekingalpha.com/"
}

SEEKING_ALPHA_SECTIONS = [
    "https://seekingalpha.com/news",
    "https://seekingalpha.com/market-news",
    "https://seekingalpha.com/earnings/earnings-news"
]

def setup_driver():
    """Setup Selenium driver for seeking alpha (handles JSè-heavy content)"""
    options=Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_arguemnt("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(f"--user-agent={headers['User-Agent']}")
    return webdriver.Chrome(options=options)

def fetch_seeking_alpha_article_links_selenium(max_articles=50):
    """Fetch Seeking Alpha article links using Selenium (for JS content)"""
    driver=setup_driver()
    all_links=[]
    for section_url in SEEKING_ALPHA_SECTIONS:
        try:
            print(f"Scraping {section_url}")
            driver.get(section_url)
            WebDriverWait(driver,10).until(
                EC.presence_of_element_located((By.TAG_NAME,"article"))
            )

            time.sleep(3)
            soup=BeautifulSoup(driver.page_source,'html.parser')

            link_selectors=[
                'article h3 a',
                'a[data-test-id="post-list-item-title"]',
                'h3 a[href*="/article/"]',
                'h3 a[href*="/news/"]',
                '.media-body h3 a',
                'article .title a'
            ]
            for selector in link_selectors:
                elements=soup.select(selector)
                for elem in elements:
                    href = elem.get("href", "")
                    title = elem.get_text(strip=True)
                    
                    if href and title and len(title) > 25:
                        # Ensure full URL
                        if href.startswith("/"):
                            href = f"https://seekingalpha.com{href}"
                        elif not href.startswith("http"):
                            continue
                            
                        # Filter for relevant articles
                        if ("seekingalpha.com" in href and 
                            ("/article/" in href or "/news/" in href) and
                            title not in [link[0] for link in all_links]):
                            all_links.append((title, href))
            
            time.sleep(2)  # Rate limiting between sections
            
        except Exception as e:
            print(f"[!] Error scraping {section_url}: {e}")
            continue
    
    driver.quit()
    
    # Remove duplicates and limit
    unique_links = list(dict.fromkeys(all_links))[:max_articles]
    print(f"[✓] {len(unique_links)} Seeking Alpha links found")
    return unique_links

def fetch_seeking_alpha_article_links_requests(max_articles=50):
    """Fallback method using requests (may have limited success due to JS)"""
    all_links = []
    
    for section_url in SEEKING_ALPHA_SECTIONS:
        try:
            print(f"[📰] Scraping {section_url} (requests method)...")
            response = requests.get(section_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Basic selectors that might work without JS
            link_selectors = [
                'h3 a[href*="/article/"]',
                'h3 a[href*="/news/"]',
                'a[href*="/article/"]',
                '.title a'
            ]
            
            for selector in link_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    href = elem.get("href", "")
                    title = elem.get_text(strip=True)
                    
                    if href and title and len(title) > 25:
                        if href.startswith("/"):
                            href = f"https://seekingalpha.com{href}"
                        
                        if ("seekingalpha.com" in href and 
                            title not in [link[0] for link in all_links]):
                            all_links.append((title, href))
            
            time.sleep(1)
            
        except Exception as e:
            print(f"[!] Error scraping {section_url}: {e}")
            continue
    
    unique_links = list(dict.fromkeys(all_links))[:max_articles]
    print(f"[✓] {len(unique_links)} Seeking Alpha links found via requests")
    return unique_links

def extract_article_content(url):
    """Extract article content from Seeking Alpha article page"""
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Seeking Alpha content selectors
        content_selectors = [
            'div[data-test-id="content-container"] p',
            'div.paywall-full-content p',
            'div[data-test-id="article-content"] p',
            'article .content p',
            'div.article-content p',
            '.sa-art p'
        ]
        
        content = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                paragraphs = []
                for p in elements:
                    text = p.get_text(strip=True)
                    # Skip Seeking Alpha boilerplate
                    if (text and 
                        "This article was written by" not in text and
                        "Seeking Alpha" not in text and
                        "Follow" not in text and
                        "Disclosure:" not in text and
                        len(text) > 20):
                        paragraphs.append(text)
                
                if paragraphs:
                    content = " ".join(paragraphs)
                    break
        
        # Alternative approach for paywalled content preview
        if not content:
            # Look for preview/summary content
            preview_selectors = [
                'div.summary p',
                'div.article-summary p',
                '.lead-summary',
                'div[data-test-id="post-summary"]'
            ]
            
            for selector in preview_selectors:
                elements = soup.select(selector)
                if elements:
                    content = " ".join(elem.get_text(strip=True) for elem in elements)
                    break
        
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
    print(f"[✓] {len(df)} Seeking Alpha articles saved to database")

def main(use_selenium=True):
    """Main scraping pipeline for Seeking Alpha"""
    articles = []
    
    # Try Selenium first for better JS support, fallback to requests
    if use_selenium:
        try:
            links = fetch_seeking_alpha_article_links_selenium()
        except Exception as e:
            print(f"[!] Selenium failed: {e}, falling back to requests...")
            links = fetch_seeking_alpha_article_links_requests()
    else:
        links = fetch_seeking_alpha_article_links_requests()
    
    print(f"Processing {len(links)} Seeking Alpha articles...")
    
    for title, url in links:
        print(f"Scraping: {title[:60]}...")
        content = extract_article_content(url)
        time.sleep(2.5)  # Conservative rate limiting for Seeking Alpha
        
        if len(content.strip()) < 200:  # Lower threshold for SA due to potential paywalls
            print(f"[!] Content too short for: {title[:40]}...")
            continue
            
        articles.append({
            "title": title,
            "content": content,
            "summary": None,
            "url": url,
            "source": "Seeking Alpha",
            "date": pd.Timestamp.now().strftime("%Y-%m-%d")
        })
    
    if articles:
        df = pd.DataFrame(articles)
        save_articles_to_db(df)
        print(f"[✅] Seeking Alpha scraping completed: {len(articles)} articles")
    else:
        print("[!] No valid Seeking Alpha articles found")

if __name__ == "__main__":
    main()