import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import time
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
# ✅ Config
headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.google.com/"
}

FT_URL = "https://www.ft.com/technology"

# Étape 1 : collecter les articles
def fetch_ft_article_links():
    response = requests.get(FT_URL, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")  # type: ignore
        title = a.get_text(strip=True)
        if "/content/" in str(href) and len(title) > 30:
            full_url = "https://www.ft.com" + str(href)
            links.append((title, full_url))

    unique_links = list(dict.fromkeys(links))[:5]
    return unique_links

# Étape 2 : extraire le contenu d’un article
def extract_article_content(url):
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")

        content = " ".join(p.get_text(strip=True) for p in paragraphs)
        return content

    except Exception as e:
        print(f"[!] Erreur avec {url} : {e}")
        return ""

# ✅ Étape 3 : sauvegarder dans SQLite
def save_articles_to_db(df, db_path=os.path.join(DATA_DIR, "articles.db")):
    conn = sqlite3.connect(db_path)
    df.to_sql("articles", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"[✓] {len(df)} articles FT enregistrés dans la base.")

# ✅ Main pipeline
def main():
    articles = []
    links = fetch_ft_article_links()
    print(f"{len(links)} liens extraits depuis Financial Times.")

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
        print("Aucun article complet valide trouvé.")

if __name__ == "__main__":
    main()
