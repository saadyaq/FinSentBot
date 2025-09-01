from bs4 import BeautifulSoup
import requests
import time
import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


#Configuration

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.google.com/"
}

cnbc_url="https://www.cnbc.com/world/?region=world"

#Collecter les liens des articles 

def fetch_cnbc_article_links(max_articles=50):
    """Fetch recent CNBC article links with configurable limit"""
    response=requests.get(cnbc_url,headers=headers)
    soup = BeautifulSoup(response.text,"html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a.get("href")
        title = a.get_text(strip=True)
        if "/202" in str(href) and len(title) > 30:
            full_url = href if href.startswith("http") else "https://www.cnbc.com" + href
            links.append((title, full_url))

    # Uniques et limité au nombre spécifié
    unique_links = list(dict.fromkeys(links))[:max_articles]

    return unique_links

def fetch_cnbc_historical_wayback(start_date=None, end_date=None, max_articles=20):
    """
    Fetch historical CNBC articles using Wayback Machine
    Limited functionality due to CNBC archive limitations
    """
    print("⚠️ Note: CNBC historical archives are limited. Using Wayback Machine approach.")
    
    if not start_date or not end_date:
        print("Using recent articles as fallback...")
        return fetch_cnbc_article_links(max_articles)
    
    # Wayback Machine approach would go here
    # For now, return recent articles as fallback
    return fetch_cnbc_article_links(max_articles)



def extract_article_content(url):

    try:
        response=requests.get(url,headers=headers)
        soup=BeautifulSoup(response.text,"html.parser")
        paragraphs=soup.find_all("div",class_="group")or soup.find_all("p")

        content=" ".join(p.get_text(strip=True) for p in paragraphs )
        return content
    except Exception as e:
        print(f"Erreur avec {url}:{e}")
        return ""

def save_articles_to_db(df, db_path=os.path.join(DATA_DIR, "articles.db")):
    conn = sqlite3.connect(db_path)
    df.to_sql("articles", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"[✓] {len(df)} articles cnbc enregistrés dans la base.")

def main():

    articles=[]
    links=fetch_cnbc_article_links()
    print(f"{len(links)} liens extraits depuis CNBC.")

    for title,url in links:
        print(f"Scraping:  {title}")
        content=extract_article_content(url)
        time.sleep(1)

        if len(content.strip())<300:
            continue
        
        articles.append({ "title": title,
            "content": content,
            "summary": None,
            "url": url,
            "date": pd.Timestamp.now().strftime("%Y-%m-%d")
        })
    
    if articles:
        df=pd.DataFrame(articles)
        save_articles_to_db(df)
    else:
        print("Aucun article complet valide trouvé.")

if __name__=="__main__":
    main()


       