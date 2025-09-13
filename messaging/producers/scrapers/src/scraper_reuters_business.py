import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import time 
import json 
import os 

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR= os.path.join(BASE_DIR,"data")
os.makedirs(DATA_DIR,exist_ok=True)


# Configuration
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.reuters.com/business/"
}


REUTERS_BUSINESS_URL = "https://www.reuters.com/business/"
REUTERS_API_URL = "https://www.reuters.com/pf/api/v3/content/fetch/articles-by-section-alias"

def fetch_reuters_article_links(max_articles=50):
    """ Fetch Reuters Business article links using their API endpoint"""

    try:
        params={
            "query": json.dumps({
                "offset":0,
                "size":max_articles,
                "sectionAlias":"business",
                "website":"reuters"
            })
        }

        api_headers={
            **headers,
            "Accept":"application/json",
            "Referer":"https/www.reuters.com/business/"
        }

        response= requests.get(REUTERS_API_URL,params=params, headers=api_headers,timeout=10)
        if response.status_code==200:
            data=response.json()
            articles=[]
            
            for item in data.get("result",{}).get("articles",[]):
                title=item.get("title")
                canonical_url=item.get("canonical_url","")
                if title and canonical_url :
                    full_url=f"https://www.reuters.com{canonical_url}"
                    articles.append((title.strip()),full_url)

            print(f"{len(articles)} Reuters Business articles found via API")
            return max[:max_articles]
    
    except Exception as e:
        print(f"API approach failed:{e}, trying web scraping")

    
    try:
        response=response.get(REUTERS_BUSINESS_URL,headers=headers,timeout=10)
        soup=BeautifulSoup(response.text,"html.parser")
        links=[]

        article_selectors=[
            'a[data-testid="Heading"]',
            'a[data-testid="Link"]',
            'h3 a',
            '.story-title a',
            'a[href*="/business/]'
        ]

        for selector in article_selectors:
            elements=soup.select(selector)
            for elem in elements:
                href=elem.get("href","")
                title=elem.get_text(strip=True)

                if href and title and len(title) >30:
                    if href.startswith("/"):
                        href=f"https://reuters.com{href}"
                    elif not href.startswith("http"):
                        continue
                    
                    if "/business/" in href and "reuters.com" in href:
                        links.append((title,href))
                    
        unique_links=list(dict.fromkeys(links))[:max_articles]
        print(f"{len(unique_links)} Reuters Business links found via scraping")
        return unique_links
    except Exception as e :
        print(f" Error fetching Reuters links:{e}")
        return []


def extract_article_content(url):
    """Extract article content from Reuters article page"""

    