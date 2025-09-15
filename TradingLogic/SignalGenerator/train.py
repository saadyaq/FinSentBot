import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional 
import json 
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from kafka import KafkaConsumer, KafkaProducer 
import yfinance as yf


@dataclass
class TradingSignal:
    symbol:str
    action:str #Buy,Hold,Sell
    confidence:float
    price: float
    timestamp:str
    reasoning:str
    stop_loss:Optional[float]=None
    take_profit:Optional[float]=None
    position_size:Optional[float]=None

class TechnicalIndicators:
    """Calculate technical indicators for enhanced features """

    @staticmethod
    def rsi(prices:pd.Series, window:int =14) ->float:
        """Relative Strength Index"""

        if len(prices) < window +1 :
            return 50.0 #Neutral 
        
        delta= prices.diff()
        gain=(delta.where(delta>0.0)).rolling(window=window).mean()
        loss=(delta.where(delta<0.0)).rolling(window=window).mean()

        rs=gain/loss

        rsi=100 - (100/(1+rs))
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window:int=20) ->Tuple[float,float,float]:
        """ Bollinger Bands (upper,middle,low)"""
        if len(prices) < window :
            current_price=float(prices.iloc[-1])
            return current_price *1.02, current_price, current_price *0.98
        sma=prices.rolling(window=window).mean()
        std=prices.rolling(window=window).std()

        upper=sma + (std*2)
        lower= sma - (std*2)

        return float(upper.iloc[-1]), (float(sma.iloc[-1])), float(lower.iloc[-1])
    
    @staticmethod
    def macd(prices:pd.Series) ->Tuple[float,float]:
        """Macd and Signal Line"""

        if len(prices) < 26:
            return 0.0,0.0
        ma_periods=[5,10,20,50]
        mas={}

        for period in ma_periods:
            if len(prices) >= period:
                mas[f'ma_{period}'] = float(prices.rolling(window=period).mean().iloc[-1])
            else:
                mas[f'ma_{period}'] = float(prices.iloc[-1])
        
        return mas

class LSTMGenerator(nn.Module):
    """Advanced LSTM-based signal generator"""

    def __init__(self,input_dim:int,hidden_dim:int=64, num_layers:int=2, output_dim:int=3):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num__layers=num_layers
        #LSTM LAYER
        self.lstm=nn.LSTM(input_dim,hidden_dim,num_layers,batch_first=True,dropout=0.2)

        #Attention mechanism
        self.attention=nn.MultiheadAttention(hidden_dim,num_head=4, batch_first=True)

        #Output Classifier
        self.classifier=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2,output_dim)
        )

        #Confidence estimator
        self.confidence=nn.Sequential(
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()
        )
    
    def forward(self,x):

        lstm_out,_=self.lstm(x)
        attn_out, _=self.attention(lstm_out,lstm_out,lstm_out)
        last_hidden=attn_out[:,-1,:]
        logits=self.classifier(last_hidden)
        confidence=self.confidence(logits)

        return logits, confidence 
