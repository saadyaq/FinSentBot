#!/usr/bin/env python3
"""
Historical News Collection Script for FinSentBot
Collects historical news data to expand the training dataset

Usage:
    python collect_historical_news.py --start-date 2024-08-01 --end-date 2024-08-20
    python collect_historical_news.py --days-back 30 --sources techcrunch coindesk
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from messaging.producers.scrapers.src.historical_scraper import HistoricalScraper
from utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Historical News Collection for FinSentBot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect last 30 days from all sources
  python collect_historical_news.py --days-back 30
  
  # Collect specific date range from TechCrunch only
  python collect_historical_news.py --start-date 2024-08-01 --end-date 2024-08-20 --sources techcrunch
  
  # Collect last week, save to files only (no Kafka)
  python collect_historical_news.py --days-back 7 --no-kafka
        """
    )
    
    # Date options (mutually exclusive)
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--days-back", 
        type=int, 
        help="Number of days back from today to collect"
    )
    date_group.add_argument(
        "--start-date", 
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date", 
        help="End date (YYYY-MM-DD). Required if --start-date is used"
    )
    
    parser.add_argument(
        "--sources", 
        nargs="+", 
        default=["techcrunch", "coindesk"],
        choices=["techcrunch", "coindesk", "cnbc", "ft"],
        help="News sources to scrape (default: techcrunch coindesk)"
    )
    
    parser.add_argument(
        "--no-kafka", 
        action="store_true", 
        help="Save to files only, don't send to Kafka pipeline"
    )
    
    parser.add_argument(
        "--max-per-day", 
        type=int, 
        default=10,
        help="Maximum articles per day per source (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start_date and not args.end_date:
        parser.error("--end-date is required when using --start-date")
    
    # Calculate date range
    if args.days_back:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=args.days_back)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
    else:
        start_date_str = args.start_date
        end_date_str = args.end_date
        
        # Validate date format
        try:
            datetime.strptime(start_date_str, "%Y-%m-%d")
            datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError as e:
            parser.error(f"Invalid date format: {e}")
    
    # Show configuration
    logger.info("🚀 Historical News Collection Starting")
    logger.info(f"📅 Date range: {start_date_str} to {end_date_str}")
    logger.info(f"📰 Sources: {', '.join(args.sources)}")
    logger.info(f"📊 Max articles per day per source: {args.max_per_day}")
    logger.info(f"🔄 Send to Kafka: {not args.no_kafka}")
    
    # Initialize scraper
    try:
        scraper = HistoricalScraper()
        
        # Override send_to_kafka behavior if needed
        original_send = not args.no_kafka
        
        # Run collection
        articles = scraper.run_historical_collection(
            start_date=start_date_str,
            end_date=end_date_str, 
            sources=args.sources
        )
        
        # Summary
        if articles:
            logger.info("🎉 Historical collection completed successfully!")
            logger.info(f"📊 Total articles collected: {len(articles)}")
            
            # Breakdown by source
            source_counts = {}
            for article in articles:
                source = article.get("source", "Unknown")
                source_counts[source] = source_counts.get(source, 0) + 1
            
            logger.info("📈 Breakdown by source:")
            for source, count in source_counts.items():
                logger.info(f"  {source}: {count} articles")
            
            logger.info(f"💾 Files saved in: {scraper.data_dir}")
            
            if not args.no_kafka:
                logger.info("✅ Articles sent to Kafka pipeline for processing")
                logger.info("🔄 Run 'python TradingLogic/prepare_dataset.py' to update training dataset")
        else:
            logger.warning("⚠️ No articles were collected. Check your date range and sources.")
            
    except KeyboardInterrupt:
        logger.info("\n⏸️ Collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error during collection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()