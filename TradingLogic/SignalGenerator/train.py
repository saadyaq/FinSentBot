import pandas as pd 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#Dataset Pytorch

class TradingDataset(Dataset)
    def __init__(self,df):
        self.X=df[['symbol','text','sentiment_score','price_now','price_future','variation']].values
        self.y=df["action"].values

        #Encoder les labels

        self.label_encoder=LabelEncoder()
        self.y=self.label_encoder.fit_tranform(self.y)

    