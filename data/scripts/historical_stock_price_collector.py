import json
import time 
import pandas as pd # type: ignore
import yfinance as yf
from datetime import datetime
from pathlib import Path
import requests
from typing import List, Dict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

#Configuration 
BASE_DIR= Path(__file__).resolve().parent.parent
DATA_DIR= BASE_DIR/"raw"
DATA_DIR.mkdir(parents=True,exist_ok=True)

#Paramètres d'expansion historiques

Historical_periods=['1mo','3mo','6mo','1y']
# Smart interval selection based on Yahoo Finance limitations
Intervals_by_period = {
    '1mo': ['1h'],  # Only hourly for 1 month (avoid 1m/5m/15m limits)
    '3mo': ['1h'],  # Only hourly for 3 months  
    '6mo': ['1h'],  # Only hourly for 6 months
    '1y': ['1d'],   # Daily for 1 year (most reliable)
    '2y': ['1d'],   # Daily for longer periods
    '5y': ['1d']
}
max_symbols= 100
batch_size=10  # Reduced for better rate limiting
parallel_workers=3  # Reduced to avoid overwhelming API
sleep_between_batches=3.0  # Increased delay
sleep_between_requests=0.2  # Added delay between individual requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)


class HistoricalStockCollector:
    def __init__(self):
        self.sp500_symbols=self._get_extended_symbols()
        self.collected_data=[]
        # Collection statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'symbols_with_data': set(),
            'symbols_without_data': set(),
            'total_records': 0
        }
    
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
    
    def collect_symbol_historical_data(self,symbol:str,periods:List[str]=None,custom_intervals:List[str]=None) ->List[Dict]:

        """Collecte les données historiques pour un symbole donné avec gestion intelligente des intervalles"""
        if periods is None:
            periods=Historical_periods
        
        symbol_data=[]
        success_count = 0
        error_count = 0

        for period in periods:
            # Use smart interval selection or custom intervals
            intervals_to_use = custom_intervals if custom_intervals else Intervals_by_period.get(period, ['1h'])
            
            for interval in intervals_to_use:
                try:
                    self.stats['total_requests'] += 1
                    ticker=yf.Ticker(symbol)
                    max_retries=3
                    for attempt in range(max_retries):
                        try:
                            data=ticker.history(period=period,interval=interval)
                            break
                        except Exception as retry_e:
                            if attempt==max_retries -1:
                                raise retry_e
                            # Progressive backoff for retries
                            time.sleep((attempt + 1) * 0.5)
                    
                    if not data.empty:
                        records_added = 0
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
                            records_added += 1
                        
                        success_count += 1
                        self.stats['successful_requests'] += 1
                        self.stats['symbols_with_data'].add(symbol)
                        self.stats['total_records'] += records_added
                        
                        if records_added > 0:
                            logger.debug(f"✅ {symbol} {period}/{interval}: {records_added} records")
                    else:
                        logger.debug(f"⚠️ {symbol} {period}/{interval}: No data returned")

                    # Rate limiting between requests
                    time.sleep(sleep_between_requests)
                except Exception as e:
                    error_count += 1
                    self.stats['failed_requests'] += 1
                    self.stats['symbols_without_data'].add(symbol)
                    logger.warning(f"❌ {symbol} {period}/{interval}: {str(e)[:100]}...")
                    continue

        # Log success/error ratio
        total_attempts = success_count + error_count
        success_rate = (success_count / total_attempts * 100) if total_attempts > 0 else 0
        
        if len(symbol_data) > 0:
            logger.info(f"✅ {symbol}: {len(symbol_data)} records collected (Success: {success_count}/{total_attempts}, {success_rate:.1f}%)")
        else:
            logger.warning(f"⚠️ {symbol}: No data collected (Errors: {error_count}/{total_attempts})")
        
        return symbol_data
    
    def print_collection_stats(self):
        """Affiche les statistiques de collecte en temps réel"""
        if self.stats['total_requests'] == 0:
            return
            
        success_rate = (self.stats['successful_requests'] / self.stats['total_requests'] * 100)
        symbols_with_data_count = len(self.stats['symbols_with_data'])
        symbols_without_data_count = len(self.stats['symbols_without_data'])
        
        logger.info(f"📊 Collection Stats: {self.stats['successful_requests']}/{self.stats['total_requests']} requests successful ({success_rate:.1f}%)")
        logger.info(f"📈 Records: {self.stats['total_records']:,} | Symbols with data: {symbols_with_data_count} | Without data: {symbols_without_data_count}")

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

        logger.info("🚀 Starting Massive Historical Stock price collection")
        logger.info(f"📊 {len(self.sp500_symbols)} symbols * {len(Historical_periods)} periods")
        logger.info(f"🎯 Using smart interval selection based on Yahoo Finance limitations")

        all_data=[]

        def chunked(lst,n):
            for i in range(0,len(lst),n):
                yield lst[i:i+n]
        
        batch_num=0
        total_batches=len(list(chunked(self.sp500_symbols,batch_size)))

        for symbols_batch in chunked(self.sp500_symbols,batch_size):
            batch_num +=1
            logger.info(f"Processing batch {batch_num}/{total_batches}: {symbols_batch}")

            batch_data=self.collect_parallel_batch(symbols_batch)
            all_data.extend(batch_data)

            logger.info(f"Batch {batch_num} completed: {len(batch_data)} records")
            logger.info(f" Total collected so far: {len(all_data):,} records")
            
            # Show collection statistics after each batch
            self.print_collection_stats()

            if batch_num%save_frequency==0:
                self._save_intermediate(all_data, f"batch_{batch_num}")
                logger.info(f"Intermediate save completed")

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

        csv_file=DATA_DIR/"historical_stock_prices.csv"
        df.to_csv(csv_file,index=False)
        logger.info(f"💾 Final dataset saved:")
        logger.info(f"📄 JSONL: {jsonl_file}")
        logger.info(f"📊 CSV: {csv_file}")
    
    def _generate_collection_report(self,df:pd.DataFrame): 
        """ Génère un rapport détaillé de la collecte"""

        report ={
            "collection_metadata":{
                "collection_data":datetime.now().isoformat(),
                "total_records": len(df),
                "unique_symbols":df["symbol"].unique().tolist(),
                "periods_covered": sorted(df['period'].unique().tolist()),
                "intervals_covered": sorted(df['interval'].unique().tolist()),
                "data_range":{
                    "earliest":df['timestamp'].min() if not df.empty else None,
                    "latest": df['timestamp'].max() if not df.empty else None

                }

            },
            "data_quality":{
                "records_per_symbol": df.groupby('symbol').size().describe().to_dict(),
                "records_per_period":df['period'].value_counts().to_dict(),
                "records_per_interval":df['interval'].value_counts().to_dict(),
                "price_range":{
                    "min":float(df["price"].min()) if not df.empty else 0,
                    "max": float(df["price"].max()) if not df.empty else 0,
                    "mean": float(df['price'].mean()) if not df.empty else 0
                }

            },
            "collection_performance":{
                "symbols_requested": len(self.sp500_symbols),
                "symbols_with_data":df['symbol'].nunique(),
                "success_rate": df['symbol'].nunique()/len(self.sp500_symbols),
                "avg_records_per_symbol": len(df)/ max(df['symbol'].nunique(), 1),
                "api_stats": {
                    "total_requests": self.stats['total_requests'],
                    "successful_requests": self.stats['successful_requests'],
                    "failed_requests": self.stats['failed_requests'],
                    "api_success_rate": (self.stats['successful_requests'] / max(self.stats['total_requests'], 1) * 100)
                }
            }

            
        }
        report_file=DATA_DIR/ f"historical_data_collection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file,'w') as f:
            json.dump(report,f)

        # Affichage console
        print("\n" + "="*70)
        print("📊 HISTORICAL STOCK PRICE COLLECTION REPORT")
        print("="*70)
        print(f"🎯 Total Records: {report['collection_metadata']['total_records']:,}")
        print(f"🏢 Unique Symbols: {report['collection_metadata']['unique_symbols']}")
        print(f"📅 Periods: {', '.join(report['collection_metadata']['periods_covered'])}")
        print(f"⏱️  Intervals: {', '.join(report['collection_metadata']['intervals_covered'])}")
        print(f"💰 Price Range: ${report['data_quality']['price_range']['min']:.2f} - ${report['data_quality']['price_range']['max']:.2f}")
        print(f"✅ Symbol Success Rate: {report['collection_performance']['success_rate']:.1%}")
        print(f"🔗 API Success Rate: {report['collection_performance']['api_stats']['api_success_rate']:.1f}%")
        print(f"📈 Avg Records/Symbol: {report['collection_performance']['avg_records_per_symbol']:.0f}")
        print(f"📊 API Requests: {report['collection_performance']['api_stats']['total_requests']} ({report['collection_performance']['api_stats']['successful_requests']} successful)")
        print("="*70)
        print(f"📁 Report saved to: {report_file}")


    def get_symbols_from_existing_news(self) -> List[str]:
        """Récupère les symboles déjà présents pour focuser sur ceux-la"""
        news_file = DATA_DIR / "news_sentiment.jsonl"
        
        if not news_file.exists():
            logger.info(f"📰 News file not found: {news_file}")
            logger.info("🔄 Will use S&P 500 symbols instead")
            return []

        symbols = set()
        try:
            lines_processed = 0
            with open(news_file, 'r', encoding='utf-8') as f:
                for line in f:
                    lines_processed += 1
                    try:
                        data = json.loads(line.strip())
                        if 'symbol' in data and data['symbol']:
                            symbols.add(data['symbol'].upper().strip())
                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue
            
            symbols_list = list(symbols)
            logger.info(f"📊 Processed {lines_processed} news entries")
            logger.info(f"🎯 Found {len(symbols_list)} unique symbols in existing news")
            return symbols_list

        except FileNotFoundError:
            logger.info(f"📰 News file not accessible: {news_file}")
            return []
        except Exception as e:
            logger.warning(f"⚠️ Error reading news file: {str(e)[:100]}...")
            logger.info("🔄 Will use S&P 500 symbols instead")
            return []

    def run_focused_collection(self):
        """ Lance une collecte focalisée sur les symboles des news existantes"""

        # Récupérer les symboles de tes news
        news_symbols = self.get_symbols_from_existing_news()
        
        if news_symbols:
            logger.info(f"🎯 FOCUSED collection on {len(news_symbols)} symbols from existing news")
            self.sp500_symbols = news_symbols[:50]  # Limiter pour commencer
        else:
            logger.info("📈 Using S&P 500 symbols (no existing news found)")
        
        # Lancer la collecte
        return self.run_massive_collection()


if __name__ == "__main__":
    collector = HistoricalStockCollector()
    
    print("Choose collection mode:")
    print("1. Focused collection (symbols from your existing news)")
    print("2. Full S&P 500 collection") 
    print("3. Custom symbol list")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        collector.run_focused_collection()
    elif choice == "2":
        collector.run_massive_collection()
    elif choice == "3":
        custom_symbols = input("Enter symbols separated by commas: ").strip().upper().split(',')
        custom_symbols = [s.strip() for s in custom_symbols if s.strip()]
        collector.sp500_symbols = custom_symbols
        collector.run_massive_collection()
    else:
        print("Invalid choice, running focused collection...")
        collector.run_focused_collection()

































   