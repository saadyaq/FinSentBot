#!/usr/bin/env python3
"""
Test script for HistoricalStockCollector
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data.scripts.historical_stock_price_collector import HistoricalStockCollector
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_symbol_fetching():
    """Test S&P 500 symbol fetching"""
    logger.info("Testing S&P 500 symbol fetching...")
    collector = HistoricalStockCollector()
    symbols = collector._get_extended_symbols()
    
    print(f"✅ Fetched {len(symbols)} S&P 500 symbols")
    print(f"First 10 symbols: {symbols[:10]}")
    return len(symbols) > 0

def test_single_symbol_collection():
    """Test collecting data for a single symbol"""
    logger.info("Testing single symbol data collection...")
    collector = HistoricalStockCollector()
    
    # Test with AAPL for 1 month, 1 hour interval (smaller dataset)
    test_data = collector.collect_symbol_historical_data(
        "AAPL", 
        periods=['1mo'], 
        custom_intervals=['1h']
    )
    
    print(f"✅ Collected {len(test_data)} records for AAPL")
    if test_data:
        print(f"Sample record: {test_data[0]}")
    return len(test_data) > 0

def test_news_symbols():
    """Test getting symbols from existing news"""
    logger.info("Testing news symbol extraction...")
    collector = HistoricalStockCollector()
    
    news_symbols = collector.get_symbols_from_existing_news()
    print(f"✅ Found {len(news_symbols)} symbols from existing news")
    if news_symbols:
        print(f"First 10 news symbols: {news_symbols[:10]}")
    return True

def test_batch_collection():
    """Test small batch collection"""
    logger.info("Testing batch collection with 3 symbols...")
    collector = HistoricalStockCollector()
    
    # Override with small test set
    collector.sp500_symbols = ["AAPL", "MSFT", "GOOGL"]
    
    # Test batch processing
    batch_data = collector.collect_parallel_batch(collector.sp500_symbols)
    print(f"✅ Batch collected {len(batch_data)} total records")
    return len(batch_data) > 0

def main():
    """Run all tests"""
    print("🧪 Running HistoricalStockCollector Tests")
    print("=" * 50)
    
    tests = [
        ("Symbol Fetching", test_symbol_fetching),
        ("Single Symbol Collection", test_single_symbol_collection),
        ("News Symbols Extraction", test_news_symbols),
        ("Batch Collection", test_batch_collection),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n🔍 Running: {test_name}")
            success = test_func()
            results.append((test_name, success))
            print(f"{'✅' if success else '❌'} {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check the output above")

if __name__ == "__main__":
    main()