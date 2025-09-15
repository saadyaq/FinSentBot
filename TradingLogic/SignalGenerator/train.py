import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from .models import LSTMSignalGenerator, SimpleMLP


class TradingDataset(Dataset):
    """Dataset Pytorch pour les données de trading """
    def __init__(self, dataframe:pd.DataFrame, sequence_length:int=20,use_sequences:bool=True):
        """
        Args:
            dataframe: DataFrame avec colonnes features + 'action'
            sequence_length: Longueur des séquences pour LSTM
            use_sequences: Si False, utilise juste les features individuelles
        """

        self.sequence_length=sequence_length
        self.use_sequences=use_sequences

        feature_cols=['sentiment_score','price_now','price_future','variation']
        self.label_encoder=LabelEncoder()
        self.y=self.label_encoder.fit_transform(dataframe['action'].values)
        self.scaler=StandardScaler()
        self.X=self.scaler.fit_tranform()
        print(f"Dataset créé: {len(self.X)} échantillons")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(f"Distribution: {np.bincount(self.y)}")
    