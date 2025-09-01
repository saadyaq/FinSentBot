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
        ""