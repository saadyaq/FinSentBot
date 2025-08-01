import pandas as pd 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

#Dataset Pytorch

class TradingDataset(Dataset)
    def __init__(self,df):
        self.X=df[['sentiment_score','price_now','price_future','variation']].values.astype(np.float32)
        self.y=df["action"].values

        #Encoder les labels

        self.label_encoder=LabelEncoder()
        self.y=self.label_encoder.fit_tranform(df["action"])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return torch.tensor(self.X[idx],dtype=torch.float32),torch.tensor(self.y[idx],dtype=torch.long)

# Modèle MLP

class TradingMLP(nn.Module):
    def __init__(self,input_dim, hidden_dim=16,output_dim=3):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.Relu(),
            nn.Linear(hidden_dim,output_dim)
        )
    def forward (self,x):
        return self.net(x)