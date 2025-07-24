import pandas as pd
from datetime import datetime , timedelta
import os

# Répertoires où se trouvent les données 

NEWS_PATH="/home/saadyaq/SE/Python/finsentbot/data/raw/news_sentiment.jsonl"
PRICES_PATH="/home/saadyaq/SE/Python/finsentbot/data/raw/stock_prices.jsonl"

#Fenetre d'observation après la news

OBSERVATION_WINDOW_MINUTES= 10
THRESHOLD= 0.005

def load_data():
    news_df=pd.read_json(NEWS_PATH,lines=True)
    prices_df=pd.read_html(PRICES_PATH, lines=True)