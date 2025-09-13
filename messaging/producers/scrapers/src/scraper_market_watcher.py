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
    """Fetch MarketWatch article links from mutliple sections"""

    all_links=[]
    for base_url in MARKETWATCH_URLS:
        try:
            print(f"Scraping{base_url}")
            response=requests.get(base_url,headers=headers,timeout=10)
            soup=BeautifulSoup(response.text,"html-parser")

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
            print(f"Error scraping {base_url}:e")
            continue

def extract_article_content(url):
    """Extract article content from MarketWtach article page"""

    