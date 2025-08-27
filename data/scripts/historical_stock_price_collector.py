import json
import time 
import pandas as pd # type: ignore
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import requests
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

#Configuration 
BASE_DIR= Path(__file__).resolve().parent.parent
DATA_DIR= BASE_DIR/"raw"
DATA_DIR.mkdir(parents=True,exist_ok=True)

#Paramètres d'expansion historiques

Historical_periods=['1mo','3mo','6mo','1y']
Intervals=['1m','5m','30m','1h']
max_symbols= 100
batch_size=20
parallel_workers=5
sleep_between_batches=2.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)


class HistoricalStockCollector:
    def __init__(self):
        self.sp500_symbols=self._get_extended_symbols()
        self.collected_data=[]
    
    def _get_extended_symbols(self) -> List[str]:
        """Récupère une liste étendue de symboles S&P 500"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            tables = pd.read_html(response.content)
            if not tables:
                raise ValueError("No tables found on Wikipedia page")
                
            df = tables[0]
            if "Symbol" not in df.columns:
                raise ValueError("Symbol column not found in Wikipedia table")
                
            sp500_symbols = df["Symbol"].astype(str).str.strip().str.upper().tolist()
            sp500_symbols = [s.replace('.', '-') for s in sp500_symbols if s and s != 'nan'][:max_symbols]
            
            logger.info(f"Loaded {len(sp500_symbols)} S&P500 symbols from Wikipedia")
            return sp500_symbols
            
        except requests.RequestException as e:
            logger.error(f"Network error fetching S&P500 symbols: {e}")
        except ValueError as e:
            logger.error(f"Data parsing error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching S&P500 symbols: {e}")
            fallback=[
                # Tech Giants
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX', 'ADBE',
                # Finance
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB',
                # Healthcare
                'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'DHR', 'BMY', 'AMGN',
                # Consumer
                'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
                # Energy
                'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'KMI', 'VLO', 'PSX', 'MPC',
                # Industrials
                'BA', 'CAT', 'GE', 'MMM', 'UPS', 'RTX', 'LMT', 'HON', 'UNP', 'FDX',
                # ETFs populaires
                'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'ARKK', 'XLF', 'XLK', 'XLE', 'XLV'
            ]
            return fallback[:max_symbols]
    
    def collect_symbol_historical_data(self,symbol:str,periods:List[str]=None,intervals :List[str]=None) ->List[Dict]:

        """Collecte les données historiques pour un symbole donné"""
        if periods is None:
            periods=Historical_periods
        if intervals is None:
            intervals=Intervals
        
        symbol_data=[]

        for period in periods:
            for interval in intervals:
                try:
                    ticker=yf.Ticker(symbol)
                    max_retries=3
                    for attempt in range(max_retries):
                        try:
                            data=ticker.history(period=period,interval=interval)
                            break
                        except Exception as retry_e:
                            if attempt==max_retries -1:
                                raise retry_e
                    
                    if not data.empty:
                        for timestamp, row in data.iterrows():
                            if pd.isna(row["Close"]) or row["Close"] <= 0:
                                continue 
                            record  ={
                                "symbol": symbol,
                                "timestamp": timestamp.isoformat(),
                                "price" : round(float(row['Close']),4),
                                "open": round(float(row["Open"]), 4),
                                "high": round(float(row["High"]), 4), 
                                "low": round(float(row["Low"]), 4),
                                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                                "period": period,
                                "interval": interval,
                                "collection_date": datetime.now().isoformat()
                            }
                            symbol_data.append(record)


                    time.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Error collecting {symbol}--{period}--{interval}: {e}")
                    continue
















































if __name__ == "__main__":
    collector = HistoricalStockCollector()
    sp500=collector._get_extended_symbols()
    print("Choose collection mode:")
    print("1. Focused collection (symbols from your existing news)")
    print("2. Full S&P 500 collection")   