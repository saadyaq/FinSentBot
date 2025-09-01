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
    