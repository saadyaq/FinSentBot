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
            
