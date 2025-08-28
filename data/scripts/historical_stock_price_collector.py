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

        logger.info(f"✅ {symbol}: {len(symbol_data)} records collected")
        return symbol_data
    

    def collect_parallel_batch(self,symbols_batch: List[str]) -> List[Dict]:
        """ Collecte en parallèle pour un lot de symbol"""

        batch_data=[]
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_symbol={
                executor.submit(self.collect_symbol_historical_data,symbol): symbol
                for symbol in symbols_batch
            }

            for future in as_completed(future_to_symbol):
                symbol=future_to_symbol[future]
                try:
                    symbol_data = future.result(timeout=120)  # Timeout 2 min par symbole
                    batch_data.extend(symbol_data)
                except Exception as e:
                    logger.error(f"Failed to collect {symbol}: {e}")
        
        return batch_data

    def run_massive_collection(self,save_frequency:int=5) -> pd.DataFrame:
        """
        Lance la collecte massive avec sauvegarde fréquente
        Args :
            save_frequency: Sauvegarder tous les N Batches 
        """

        logger.info(" Starting Massive Historical Stock price collection")
        logger.info(f"{len(self.sp500_symbols)} symbols * {len(Historical_periods)} * {len(Intervals)} intervals")
        logger.info(f"Estimated records : {len(self.sp500_symbols)* len(Historical_periods)*len(Intervals)*100:,}+")

        all_data=[]

        def chunked(lst,n):
            for i in range(0,len(lst),n):
                yield lst[i:i+1]
        
        batch_num=0
        total_batches=len(List(chunked(self.sp500_symbols,batch_size)))

        for symbols_batch in chunked(self.sp500_symbols,batch_size):
            batch_num +=1
            logger.info(f"Processing batch {batch_num}/{total_batches}: {symbols_batch}")

            batch_data=self.collect_parallel_batch(symbols_batch)
            all_data.extend(batch_data)

            logger.info(f"Batch {batch_num} completed: {len(batch_data)}records")
            logger.info(f" Total collected so far: {len(all_data):,} records")

            if batch_num%save_frequency==0:
                self._save_intermediate(all_data, f"batch_{{batch_num}}")
                logger.info(f"Intermediate save complemented")

            time.sleep(sleep_between_batches)
        
        df=pd.DataFrame(all_data)
        self._save_final_dataset(df)
        self._generate_collection_report(df)
        return df


    def _save_intermediate(self,data: List[Dict], suffix:str):
        """ Sauvegarde intermédiaire"""

        filename=f"historical_prices_intermediate_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath=DATA_DIR/filename
        with open(filepath,'w') as f:
            json.dump(data,f)
        
        logger.info(f"Intermediate save: {len(data)} records -> {filepath}")
    
    def _save_final_dataset(self, df:pd.DataFrame):

        """ Sauvegarde du dataset final"""
        jsonl_file=DATA_DIR/"historical_stock_prices.jsonl"
        df.to_json(jsonl_file,orient='records',lines=True)

        csv_file=csv_file=DATA_DIR/"historical_stock_prices.csv"
        df.to_csv(csv_file,index=False)
        logger.info(f"💾 Final dataset saved:")
        logger.info(f"📄 JSONL: {jsonl_file}")
        logger.info(f"📊 CSV: {csv_file}")
    
    













































if __name__ == "__main__":
    collector = HistoricalStockCollector()
    sp500=collector._get_extended_symbols()
    print("Choose collection mode:")
    print("1. Focused collection (symbols from your existing news)")
    print("2. Full S&P 500 collection")   