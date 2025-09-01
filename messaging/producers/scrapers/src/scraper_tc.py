import time
import sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup


import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
#Configuration

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.google.com/"
}

tc_url="https://techcrunch.com"

def fetch_tc_article_links():
    
    response=requests.get(tc_url,headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for article in soup.find_all("a", href=True):
        href = article["href"]
        title = article.get_text(strip=True)
        if href.startswith("https://techcrunch.com/") and len(title) > 30 and "202" in href:
            links.append((title, href))

    unique_links = list(dict.fromkeys(links))[:50]
    print(f"[✓] {len(unique_links)} liens extraits depuis TechCrunch.")
    return unique_links


def extract_article_content(url):

    try:
        response=requests.get(url,headers=headers)
        soup=BeautifulSoup(response.text,"html.parser")
        paragraphs=soup.find_all("p")
        content= " ".join(p.get_text(strip=True) for p in paragraphs)
        return content 
    except Exception as e:
        print(f"[!] Erreur avec {url} : {e}")
        return ""


def save_articles_to_db(df, db_path=os.path.join(DATA_DIR, "articles.db")):
    conn = sqlite3.connect(db_path)
    df.to_sql("articles", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"[✓] {len(df)} articles cnbc enregistrés dans la base.")


def main():
    
    articles = []
    links = fetch_tc_article_links()

    for title, url in links:
        print(f"Scraping: {title}")
        content = extract_article_content(url)
        time.sleep(1)

        if len(content.strip()) < 300:
            continue

        articles.append({ "title": title,
            "content": content,
            "summary": None,
            "url": url,
            "date": pd.Timestamp.now().strftime("%Y-%m-%d")
        })
    

    if articles:
        df = pd.DataFrame(articles)
        save_articles_to_db(df)
    else:
        print("[!] Aucun article valide trouvé.")

if __name__ == "__main__":
    main()
    print("[✓] Scraping TechCrunch terminé avec succès.")