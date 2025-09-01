from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import json 
import sys
#Configuration
BASE_DIR= Path(__file__).resolve().parent.parent
RAW_DIR= BASE_DIR/"raw"
Training_dir=BASE_DIR/"training_datasets"
Training_dir.mkdir(parents=True,exist_ok=True)
#Paramètres (similaires que prepare_dataset.py)

OBSERVATION_WINDOW_MINUTES=10
BUY_THRESHOLD=0.005
SELL_THRESHOLD=-0.005

class HistoricalPriceIntegrator:
    def __init__(self):
        self.news_df=None
        self.historical_prices_df=None
        self.current_prices_df=None
    
    def load_data(self):
        """Charge toutes les données possibles"""

        print("Loading all available data")

        #Charger les news avec sentiments
        news_file=RAW_DIR/"news_sentiment.jsonl"
        if news_file.exists():
            self.news_df=pd.read_json(news_file,lines=True)
            print(f"Loaded {len(self.news_df)} news with sentiment")
        else:
            print("No existing news sentiment file found ")
            return False
        
        historical_prices_file=RAW_DIR/"historical_stock_prices.jsonl"
        if historical_prices_file.exists():
            self.historical_prices_df=pd.read_json(historical_prices_file,lines=True)
            print(f"Loaded{len(self.historical_prices_df)} historical price records")
        else: 
            print("File not found run hisotrical stock price before")
        
        current_prices_file= RAW_DIR/"stock_prices.jsonl"
        if current_prices_file.exists():
            self.current_prices_df=pd.read_json(current_prices_file,lines=True)
            print(f"Loaded{len(self.current_prices_df)} current price records")
        else:
            print("No current prices file found ")
        
        return self._combine_price_data()
    
    def _combine_price_data(self):
        """ Combine les prix historiques + prix actuels"""

        prices_df=[]
        if self.historical_prices_df is not None:
            hist_prices=self.historical_prices_df[["symbol","timestamp","price"]].copy()
            hist_prices['source']="historical"
            prices_df.append(hist_prices)
        
        if self.current_prices_df is not None:
            current_prices=self.current_prices_df[['symbol','timestamp','price']].copy()
            current_prices['source']='current'
            prices_df.append(current_prices)
        
        if not prices_df:
            print("No price data available")
            return False

        self.all_prices_df=pd.concat(prices_df,ignore_index=True)
        self.all_prices_df['timestamp']=pd.to_datetime(self.all_prices_df['timestamp'], errors="coerce").dz.tz_localize(None)

        self.all_prices_df=self.all_prices_df.dropna(subset=['symbol','timestamp','price'])
        self.all_prices_df['symbol']=self.all_prices_df['symbol'].astype(str).str.upper().str.strip()
        self.all_prices_df=self.all_prices_df.sort_values("timestamp").drop_duplicates(subset=['symbol', 'timestamp'],kepp='last')
        print(f"✅ Combined price dataset: {len(self.all_prices_df):,} records")
        print(f"📊 Symbols available: {self.all_prices_df['symbol'].nunique()}")
        print(f"📅 Date range: {self.all_prices_df['timestamp'].min()} to {self.all_prices_df['timestamp'].max()}")
        
        return True
        