#!/usr/bin/env python3
"""
Historical News Scraper for FinSentBot
Collects historical articles from TechCrunch and CoinDesk archives
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from datetime import datetime, timedelta, date
from typing import List, Tuple, Dict, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kafka import KafkaProducer
from config.kafka_config import KAFKA_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)

class HistoricalScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive"
        }
        
        # Kafka producer for sending to pipeline
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_CONFIG["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            **KAFKA_CONFIG["producer_config"]
        )
        
        # Data storage paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.data_dir = os.path.join(self.base_dir, "data", "historical")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_date_range(self, start_date: str, end_date: str) -> List[date]:
        """Generate list of dates between start_date and end_date"""
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)
        
        return dates
    
    def scrape_techcrunch_historical(self, start_date: str, end_date: str, send_to_kafka: bool = True) -> List[Dict]:
        """
        Scrape TechCrunch historical articles using date-based archive URLs
        Format: techcrunch.com/YYYY/MM/DD
        """
        logger.info(f"🔍 Starting TechCrunch historical scraping: {start_date} to {end_date}")
        
        dates = self.get_date_range(start_date, end_date)
        all_articles = []
        
        for target_date in dates:
            url = f"https://techcrunch.com/{target_date.year:04d}/{target_date.month:02d}/{target_date.day:02d}"
            
            try:
                logger.info(f"📅 Scraping TechCrunch for {target_date}: {url}")
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 404:
                    logger.debug(f"No articles found for {target_date}")
                    continue
                elif response.status_code != 200:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # TechCrunch article links in archive pages
                article_links = []
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    title = link.get_text(strip=True)
                    
                    # Filter for actual article URLs
                    if (href and '/2024/' in href and 
                        len(title) > 30 and 
                        not any(skip in href for skip in ['/tag/', '/category/', '/author/', '#'])):
                        
                        full_url = href if href.startswith('http') else f"https://techcrunch.com{href}"
                        article_links.append((title, full_url))
                
                # Remove duplicates
                article_links = list(dict.fromkeys(article_links))
                logger.info(f"📰 Found {len(article_links)} articles for {target_date}")
                
                # Extract content from each article
                for title, article_url in article_links[:10]:  # Limit to 10 per day to avoid overwhelming
                    try:
                        article_content = self._extract_techcrunch_content(article_url)
                        if len(article_content.strip()) < 300:
                            continue
                            
                        article_data = {
                            "source": "TechCrunch_Historical",
                            "title": title,
                            "content": article_content,
                            "url": article_url,
                            "timestamp": datetime.combine(target_date, datetime.min.time()).isoformat(),
                            "scraped_at": datetime.utcnow().isoformat()
                        }
                        
                        all_articles.append(article_data)
                        
                        # Send to Kafka if requested
                        if send_to_kafka:
                            self.producer.send(KAFKA_CONFIG["topics"]["raw_news"], value=article_data)
                            logger.debug(f"📤 Sent to Kafka: {title[:50]}...")
                        
                        # Rate limiting
                        time.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"Error extracting content from {article_url}: {e}")
                        continue
                
                # Rate limiting between dates
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error scraping TechCrunch for {target_date}: {e}")
                continue
        
        logger.info(f"✅ TechCrunch historical scraping completed. Total articles: {len(all_articles)}")
        return all_articles
    
    def _extract_techcrunch_content(self, url: str) -> str:
        """Extract article content from TechCrunch article page"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # TechCrunch article content selectors
            content_selectors = [
                'div.article-content',
                'div.entry-content', 
                'div[data-module="ArticleBody"]',
                'div.post-content',
                '.wp-content p'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = " ".join(p.get_text(strip=True) for p in elements)
                    break
            
            # Fallback: get all paragraphs
            if not content:
                paragraphs = soup.find_all('p')
                content = " ".join(p.get_text(strip=True) for p in paragraphs)
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return ""
    
    def scrape_coindesk_historical(self, start_date: str, end_date: str, send_to_kafka: bool = True) -> List[Dict]:
        """
        Scrape CoinDesk historical articles
        Note: CoinDesk doesn't have a clean date archive, so we'll use their sitemap and RSS approaches
        """
        logger.info(f"🔍 Starting CoinDesk historical scraping: {start_date} to {end_date}")
        
        all_articles = []
        
        # Try the archive tag page first
        try:
            archive_url = "https://www.coindesk.com/tag/archives"
            response = requests.get(archive_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract article links from archive page
                article_links = []
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    title = link.get_text(strip=True)
                    
                    if (href and '/2024/' in href and 
                        len(title) > 30 and
                        'coindesk.com' in href):
                        article_links.append((title, href))
                
                logger.info(f"📰 Found {len(article_links)} potential CoinDesk articles")
                
                # Extract content from articles
                for title, article_url in article_links[:20]:  # Limit to avoid overwhelming
                    try:
                        article_content = self._extract_coindesk_content(article_url)
                        if len(article_content.strip()) < 300:
                            continue
                            
                        # Parse date from URL if possible
                        article_date = self._parse_date_from_url(article_url, start_date, end_date)
                        if not article_date:
                            continue
                            
                        article_data = {
                            "source": "CoinDesk_Historical", 
                            "title": title,
                            "content": article_content,
                            "url": article_url,
                            "timestamp": article_date,
                            "scraped_at": datetime.utcnow().isoformat()
                        }
                        
                        all_articles.append(article_data)
                        
                        if send_to_kafka:
                            self.producer.send(KAFKA_CONFIG["topics"]["raw_news"], value=article_data)
                            logger.debug(f"📤 Sent to Kafka: {title[:50]}...")
                        
                        time.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"Error extracting CoinDesk content from {article_url}: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"Error scraping CoinDesk archives: {e}")
        
        logger.info(f"✅ CoinDesk historical scraping completed. Total articles: {len(all_articles)}")
        return all_articles
    
    def _extract_coindesk_content(self, url: str) -> str:
        """Extract article content from CoinDesk article page"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # CoinDesk content selectors
            content_selectors = [
                'div.at-content-body',
                'div.entry-content',
                'div.article-content',
                'div[data-module="ArticleBody"]'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = " ".join(p.get_text(strip=True) for p in elements)
                    break
            
            # Fallback
            if not content:
                paragraphs = soup.find_all('p')
                content = " ".join(p.get_text(strip=True) for p in paragraphs)
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting CoinDesk content from {url}: {e}")
            return ""
    
    def _parse_date_from_url(self, url: str, start_date: str, end_date: str) -> Optional[str]:
        """Parse date from URL and check if it's in our target range"""
        import re
        
        # Look for date patterns in URL: /YYYY/MM/DD/
        date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
        if date_match:
            year, month, day = date_match.groups()
            article_date = f"{year}-{month}-{day}"
            
            # Check if date is in our range
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
            article_dt = datetime.strptime(article_date, "%Y-%m-%d").date()
            
            if start_dt <= article_dt <= end_dt:
                return datetime.combine(article_dt, datetime.min.time()).isoformat()
        
        return None
    
    def save_to_file(self, articles: List[Dict], filename: str):
        """Save articles to JSONL file"""
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        logger.info(f"💾 Saved {len(articles)} articles to {filepath}")
    
    def run_historical_collection(self, start_date: str, end_date: str, sources: List[str] = None):
        """Main method to run historical collection"""
        if sources is None:
            sources = ["techcrunch", "coindesk"]
        
        all_articles = []
        
        logger.info(f"🚀 Starting historical news collection: {start_date} to {end_date}")
        logger.info(f"📰 Sources: {', '.join(sources)}")
        
        if "techcrunch" in sources:
            tc_articles = self.scrape_techcrunch_historical(start_date, end_date)
            all_articles.extend(tc_articles)
            self.save_to_file(tc_articles, f"techcrunch_historical_{start_date}_to_{end_date}.jsonl")
        
        if "coindesk" in sources:
            cd_articles = self.scrape_coindesk_historical(start_date, end_date)
            all_articles.extend(cd_articles)
            self.save_to_file(cd_articles, f"coindesk_historical_{start_date}_to_{end_date}.jsonl")
        
        # Save combined results
        if all_articles:
            self.save_to_file(all_articles, f"historical_news_{start_date}_to_{end_date}.jsonl")
            logger.info(f"🎉 Historical collection complete! Total articles: {len(all_articles)}")
        else:
            logger.warning("⚠️ No historical articles collected")
        
        return all_articles


def main():
    """CLI interface for historical scraping"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Historical News Scraper for FinSentBot")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--sources", nargs="+", default=["techcrunch", "coindesk"], 
                       help="Sources to scrape (techcrunch, coindesk)")
    parser.add_argument("--no-kafka", action="store_true", help="Skip sending to Kafka")
    
    args = parser.parse_args()
    
    scraper = HistoricalScraper()
    articles = scraper.run_historical_collection(
        args.start_date, 
        args.end_date, 
        args.sources
    )
    
    print(f"\n✅ Collected {len(articles)} historical articles")
    print(f"📁 Files saved in: {scraper.data_dir}")


if __name__ == "__main__":
    main()