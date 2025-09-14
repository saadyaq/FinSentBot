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
    """Setup Selenium driver for seeking alpha (handles JS-heavy content)"""
    options=Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-web-security")
    options.add_argument("--disable-features=VizDisplayCompositor")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-images")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(f"--user-agent={headers['User-Agent']}")
    try:
        return webdriver.Chrome(options=options)
    except Exception as e:
        print(f"[!] Failed to initialize Chrome driver: {e}")
        return None

def fetch_seeking_alpha_article_links_selenium(max_articles=50):
    """Fetch Seeking Alpha article links using Selenium (for JS content)"""
    driver = setup_driver()
    if not driver:
        print("[!] Failed to setup driver, falling back to requests method")
        return fetch_seeking_alpha_article_links_requests(max_articles)
    
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
    
    try:
        driver.quit()
    except Exception as e:
        print(f"[!] Error closing driver: {e}")
    
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

def extract_article_content(url, debug=False):
    """Extract article content from Seeking Alpha article page"""
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Updated Seeking Alpha content selectors for 2025
        content_selectors = [
            'div[data-test-id="content-container"]',
            'div[data-test-id="article-content"]', 
            'div.paywall-full-content',
            'article[data-test-id="post"] div',
            'div.article-content',
            'section[data-test-id="content-detail"]',
            '[data-test-id="post-content"]',
            'div.content-detail',
            '.article-body',
            '.paywall-free-content',
            'div.summary',
            'div[data-test-id="post-summary"]'
        ]
        
        content = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                if debug:
                    print(f"[DEBUG] Found {len(elements)} elements with selector: {selector}")
                
                # Try to extract all text from the container
                for element in elements:
                    # Get all paragraph text within the container
                    paragraphs = element.find_all(['p', 'div'], recursive=True)
                    if not paragraphs:
                        # If no paragraphs found, get direct text
                        text = element.get_text(strip=True)
                        if text and len(text) > 50:
                            paragraphs = [text]
                    
                    valid_paragraphs = []
                    for p in paragraphs:
                        if hasattr(p, 'get_text'):
                            text = p.get_text(strip=True)
                        else:
                            text = str(p).strip()
                            
                        # Skip Seeking Alpha boilerplate and short content
                        if (text and 
                            len(text) > 20 and
                            "This article was written by" not in text and
                            "Seeking Alpha" not in text and
                            "Follow" not in text and
                            "Disclosure:" not in text and
                            "Editor's Note:" not in text and
                            "Click to enlarge" not in text):
                            valid_paragraphs.append(text)
                    
                    if valid_paragraphs:
                        content = " ".join(valid_paragraphs)
                        if debug:
                            print(f"[DEBUG] Content extracted: {len(content)} chars")
                        break
                
                if content:
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

def main(use_selenium=True, debug=False):
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
        content = extract_article_content(url, debug=debug)
        time.sleep(2.5)  # Conservative rate limiting for Seeking Alpha
        
        if len(content.strip()) < 50:  # Lowered threshold for SA due to paywalls
            print(f"[!] Content too short ({len(content)} chars) for: {title[:40]}...")
            # Try to get summary or preview content if main content fails
            try:
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Look for article summary or meta description
                summary_sources = [
                    soup.find("meta", {"name": "description"}),
                    soup.find("meta", {"property": "og:description"}),
                    soup.select_one('div[data-test-id="post-summary"]'),
                    soup.select_one('.summary')
                ]
                
                fallback_content = ""
                for source in summary_sources:
                    if source:
                        if hasattr(source, 'get'):
                            fallback_content = source.get('content', '')
                        else:
                            fallback_content = source.get_text(strip=True)
                        
                        if fallback_content and len(fallback_content) > 50:
                            content = fallback_content
                            print(f"[✓] Using fallback content ({len(content)} chars)")
                            break
                
                if len(content.strip()) < 50:
                    continue
                    #test
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Fallback content extraction failed: {e}")
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
    main(debug=True)  # Enable debug for testing