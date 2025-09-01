import time
import sqlite3
import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
# --- Config Selenium headless ---
def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)

# --- Étape 1 : Extraire les liens CoinDesk dynamiquement ---
def fetch_coindesk_links():
    driver = setup_driver()
    driver.get("https://www.coindesk.com/markets/")
    time.sleep(5)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    links = []
    for a in soup.select("a.text-color-charcoal-900.mb-4.hover\\:underline"):
        href = a.get("href")
        title = a.get_text(strip=True)
        if href and title and href.startswith("/") and len(title) > 20:
            full_url = "https://www.coindesk.com" + href
            links.append((title, full_url))

    unique_links = list(dict.fromkeys(links))[:50]
    return unique_links

# --- Étape 2 : Extraire le contenu d’un article ---
headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

def extract_article_content(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text(strip=True) for p in paragraphs)
        return content
    except Exception as e:
        print(f"[!] Erreur avec {url} : {e}")
        return ""

# --- Étape 3 : Sauvegarder dans la base SQLite ---
def save_articles_to_db(df, db_path=os.path.join(DATA_DIR, "articles.db")):
    conn = sqlite3.connect(db_path)
    df.to_sql("articles", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"[✓] {len(df)} articles CoinDesk enregistrés dans la base.")

# --- Pipeline principal ---
def main():
    links = fetch_coindesk_links()
    print(f"{len(links)} articles CoinDesk détectés.")

    articles = []
    for title, url in links:
        print(f"Scraping: {title}")
        content = extract_article_content(url)
        time.sleep(1)
        if len(content.strip()) < 300:
            continue
        articles.append({
            "title": title,
            "content": content,
            "summary": None,
            "url": url,
            "date": pd.Timestamp.now().strftime("%Y-%m-%d")
        })


    if articles:
        df = pd.DataFrame(articles)
        save_articles_to_db(df)
    else:
        print("[!] Aucun article complet récupéré.")

if __name__ == "__main__":
    main()
