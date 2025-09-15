import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass









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
