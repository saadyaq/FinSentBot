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
    
    
    def generate_expanded_training_samples(self) :

        """Génère les échantillons avec beaucoup plus de donnée prix"""

        self.news_df['timestamp']=pd.to_datetime(self.news_df['timestamp'], errors="coerce").dt.tz_localize(None)
        self.news_df=self.news_df.dropna(subset=['symbol','timestamp'])
        self.news_df['symbol']=self.news_df['symbol'].astype(str).str.upper().str.strip()

        self.news_df=self.news_df.drop_duplicates(subset=['symbol','timestamp'],keep='first')
        print(f"Processing {len(self.news_df)} unique news articles")

        training_samples=[]
        successful_matches=[]

        symbols_with_news=self.news_df['symbol'].unique()
        print(f" Symbols with news: {len(symbols_with_news)}")

        for symbol in symbols_with_news:
            symbol_news=self.news_df[self.news_df['symbol']==symbol].sort_values('timestamp')
            symbol_prices=self.all_prices_df[self.all_prices_df['symbol']==symbol].sort_values('timestamp')

            if symbol_prices.empty:
                print(f"No prices for {symbol}")
                continue
            
            print(f"Processing {symbol}: {len(symbol_news)} news × {len(symbol_prices)} prices")

            for _,news_row in symbol_news.iterrows():
                try:

                    news_timestamp=news_row["timestamp"]
                    prices_before = symbol_prices[symbol_prices['timestamp'] <= news_timestamp]
                    if prices_before.empty:
                        continue
                    
                    # Prix le plus proche avant la news
                    closest_price_idx = prices_before['timestamp'].idxmax()
                    p0 = prices_before.loc[closest_price_idx, 'price']
                    price_timestamp = prices_before.loc[closest_price_idx, 'timestamp']
                    
                    # Vérifier que le prix n'est pas trop ancien (max 2h)
                    time_diff = (news_timestamp - price_timestamp).total_seconds() / 3600
                    if time_diff > 2:  # Plus de 2h d'écart
                        continue
                    
                    # Prix futur (après fenêtre d'observation)
                    future_timestamp = news_timestamp + timedelta(minutes=OBSERVATION_WINDOW_MINUTES)
                    prices_after = symbol_prices[symbol_prices['timestamp'] >= future_timestamp]
                    
                    if prices_after.empty:
                        continue
                    
                    # Premier prix après la fenêtre
                    p1 = prices_after.iloc[0]['price']
                    variation = (p1 - p0) / p0
                    
                    # Logique de classification (même que prepare_dataset.py)
                    if variation >= BUY_THRESHOLD:
                        action = "BUY"
                    elif variation <= SELL_THRESHOLD:
                        action = "SELL"
                    else:
                        # Utiliser le sentiment pour les variations neutres
                        sentiment = news_row.get('sentiment_score', 0)
                        if sentiment > 0.1:
                            action = "BUY"
                        elif sentiment < -0.1:
                            action = "SELL"
                        else:
                            action = "HOLD"
                    
                    training_samples.append({
                        "symbol": symbol,
                        "text": news_row.get('content', ''),
                        "sentiment_score": news_row.get('sentiment_score', 0),
                        "price_now": p0,
                        "price_future": p1,
                        "variation": variation,
                        "action": action,
                        "news_timestamp": news_timestamp.isoformat(),
                        "price_timestamp": price_timestamp.isoformat()
                    })
                    
                    successful_matches += 1
                    
                except Exception as e:
                    print(f"⚠️ Error processing sample for {symbol}: {e}")
                    continue
        
        print(f"✅ Generated {len(training_samples)} training samples ({successful_matches} successful matches)")
        
        return pd.DataFrame(training_samples)