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
