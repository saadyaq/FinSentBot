import json 
import time 
from datetime import datetime
from kafka import KafkaProducer
from config.kafka_config import KAFKA_CONFIG

# === Import of existing scrapers ===

from messaging.producers.scrapers.src.scraper_cnbc import fetch_cnbc_article_links , extract_article_content as extract_cnbc
from messaging.producers.scrapers.src.scraper_coindesk import fetch_coindesk_links, extract_article_content as extract_coindesk
from messaging.producers.scrapers.src.scraper_ft import fetch_ft_article_links , extract_article_content as extract_ft
from messaging.producers.scrapers.src.scraper_tc import fetch_tc_article_links, extract_article_content as extract_tc

# === Config Kafka producer ===

producer = KafkaProducer(
    bootstrap_servers=KAFKA_CONFIG["bootstrap_servers"],
    value_serializer= lambda v: json.dumps(v).encode("utf-8"),
    **KAFKA_CONFIG["producer_config"]
)

def scrape_and_send(source, fetch_links, extract_func):

    """
    - Récupère les leins d'articles depuis un scraper
    - Extrait le contenu de chaque article
    - Envoie chaque message dans kafka 
    """

    print(f"🔍 Scraping {source}...")

    try :

        links = fetch_links()
        print(f"[✓] {len(links)} links trouvés pour {source}")

        for title , url in links :
            content = extract_func(url)
            time.sleep(1)

            if len(content.strip()) <300:

                continue 
            
            message = {
                "source": source,
                "title" : title,
                "content" : content,
                "url" : url, 
                "timestamp" : datetime.utcnow().isoformat()
            }

            producer.send(KAFKA_CONFIG["topics"]["raw_news"], value = message)
            print(f"📤 Sent to Kafka: {title}")

    except Exception as e:
        print(f"⚠️ Error scraping {source}: {e}")

def main():

    while True:

        print("\n=== 🗞️ New Scraping Round Started ===\n")
        scrape_and_send("CNBC", fetch_cnbc_article_links, extract_cnbc)
        scrape_and_send("CoinDesk", fetch_coindesk_links, extract_coindesk)
        scrape_and_send("Financial Times", fetch_ft_article_links, extract_ft)
        scrape_and_send("TechCrunch", fetch_tc_article_links, extract_tc)
        
        print("\n✅ Scraping round complete. Sleeping for 20 minutes...\n")
        time.sleep(1200)  # toutes les 30 minutes

if __name__ == "__main__":
    main()