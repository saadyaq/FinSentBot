import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import time
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Configuration
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.fool.com/"
}

MOTLEY_FOOL_SECTIONS = [
    "https://www.fool.com/investing/",
    "https://www.fool.com/investing/stock-market/",
    "https://www.fool.com/earnings/",
    "https://www.fool.com/recent-headlines/"
]

def fetch_motley_fool_article_links(max_articles=50):
    """Fetch The Motley Fool article links from multiple sections"""
    all_links = []
    
    for section_url in MOTLEY_FOOL_SECTIONS:
        try:
            print(f"[📰] Scraping {section_url}...")
            response = requests.get(section_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Motley Fool article selectors
            link_selectors = [
                'h2 a[href*="/investing/"]',
                'h3 a[href*="/investing/"]',
                'h4 a[href*="/investing/"]',
                'a.text-gray-1100[href*="/investing/"]',
                'article h2 a',
                'article h3 a',
                '.headlineText a',
                'a[data-track-module="ArticleLink"]'
            ]
            
            for selector in link_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    href = elem.get("href", "")
                    title = elem.get_text(strip=True)
                    
                    if href and title and len(title) > 25:
                        # Ensure full URL
                        if href.startswith("/"):
                            href = f"https://www.fool.com{href}"
                        elif not href.startswith("http"):
                            continue
                            
                        # Filter for investing articles
                        if ("fool.com" in href and 
                            "/investing/" in href and
                            title not in [link[0] for link in all_links]):
                            all_links.append((title, href))
            
            time.sleep(1.5)  # Rate limiting between sections
            
        except Exception as e:
            print(f"[!] Error scraping {section_url}: {e}")
            continue
    
    # Remove duplicates and limit
    unique_links = list(dict.fromkeys(all_links))[:max_articles]
    print(f"[✓] {len(unique_links)} Motley Fool links found")
    return unique_links

def extract_article_content(url):
    """Extract article content from Motley Fool article page"""
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Motley Fool content selectors
        content_selectors = [
            'div.tailwind-article-body p',
            'div.article-body p',
            'div.entry-content p',
            'article .article-wrap p',
            'div[data-module="ArticleBody"] p',
            '.usmf-wrapper p',
            'main article p'
        ]
        
        content = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                paragraphs = []
                for p in elements:
                    text = p.get_text(strip=True)
                    # Skip Motley Fool boilerplate and ads
                    if (text and 
                        "The Motley Fool has" not in text and
                        "Stock Advisor" not in text and
                        "recommendations" not in text and
                        "Fool.com" not in text and
                        "disclosure policy" not in text and
                        "investing advice" not in text and
                        len(text) > 20):
                        paragraphs.append(text)
                
                if paragraphs:
                    content = " ".join(paragraphs)
                    break
        
        # Alternative approach for different layouts
        if not content:
            # Look for main article content
            main_content = soup.find("main") or soup.find("article")
            if main_content:
                paragraphs = []
                for p in main_content.find_all("p"):
                    text = p.get_text(strip=True)
                    if (text and 
                        len(text) > 20 and
                        "The Motley Fool" not in text and
                        "Stock Advisor" not in text):
                        paragraphs.append(text)
                
                if paragraphs:
                    content = " ".join(paragraphs[:15])  # Limit to first 15 paragraphs
        
        # Clean up content
        content = re.sub(r'\s+', ' ', content)  # Multiple whitespace
        content = content.replace("More From The Motley Fool", "")
        content = content.replace("10 stocks we like better than", "")
        
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
    print(f"[✓] {len(df)} Motley Fool articles saved to database")

def main():
    """Main scraping pipeline for The Motley Fool"""
    articles = []
    links = fetch_motley_fool_article_links()
    print(f"Processing {len(links)} Motley Fool articles...")
    
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
            "source": "The Motley Fool",
            "date": pd.Timestamp.now().strftime("%Y-%m-%d")
        })
    
    if articles:
        df = pd.DataFrame(articles)
        save_articles_to_db(df)
        print(f"[✅] Motley Fool scraping completed: {len(articles)} articles")
    else:
        print("[!] No valid Motley Fool articles found")

if __name__ == "__main__":
    main()