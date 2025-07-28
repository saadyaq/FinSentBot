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

    #Convertir en datetime
    news_df["timestamp"]=pd.to_datetime(news_df["timestamp"])
    prices_df["timestamp"]=pd.to_datetime(prices_df["timestamp"])

    return news_df, prices_df

def generate_labels(news_df,prices_df):
    data=[]
    for _,row in news_df.iterrows():
        timestamp=row["timestamp"]
        symbol=row["symbol"]
        sentiment=row.get("sentiment_score",0)
        text=row["text"]

        #prix courant au moment de l'info
        price_now=prices_df[(prices_df["symbol"]==symbol)& (prices_df["timestamp"]<=timestamp)].sort_values("timestamp").tail(1)

        #prix après 10 minutes

        price_future=prices_df[(prices_df["symbol"]==symbol) & (prices_df["timestamp"]> timestamp + timedelta(minutes=OBSERVATION_WINDOW_MINUTES))].sort_values("timestamp").head(1)
        
        if price_now.empty or price_future.empty:
            continue

        p0= price_now.iloc[0]["price"]
        p1=price_future.iloc[1]["price"]
        variation=(p1-p0)/p0

        if variation > THRESHOLD:
            action="BUY"
        elif variation < -THRESHOLD:
            action="SELL"
        else :
            action="HOLD"
        
        data.append({
            "symbol":symbol,
            "text":text,
            "sentiment_score":sentiment,
            "price_now":p0,
            "price_future":p1,
            "variation":variation,
            "action":action               

        })

        return pd.DataFrame(data)
    
    if __name__=="__main__":
        news_df,prices_df=load_data()
        train_dataset=generate_labels(news_df,prices_df)

        os.makedirs("data/traininig_datasets",exist_ok=True)
        train_dataset.to_csv("data/training_datasets/train.csv",index=False)
        print("Dataset generated with", len(train_dataset),"samples.")