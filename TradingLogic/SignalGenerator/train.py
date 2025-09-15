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
        