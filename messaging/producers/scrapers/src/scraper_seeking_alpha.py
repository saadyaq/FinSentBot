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


