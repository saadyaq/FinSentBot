import pandas as pd
import torch
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
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        features=self.X[idx]
        label=self.y[idx]

        if self.use_sequences:
            # Répète les features pour créer une séquence
            # TODO: Utiliser de vraies séquences temporelles
            sequence = np.tile(features, (self.sequence_length, 1))
            return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        else:
            return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class ModelTrainer:
    """Gestionaire d'entrainement des modèles"""

    def __init__(self,model_type:str='lstm',device:str="auto"):
        """
        Args:
            model_type: "lstm" ou "mlp"
            device: "cpu", "cuda" ou "auto"
        """
        self.model_type = model_type
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else 
            device if device != "auto" else "cpu"
        )
        
        print(f"Entraînement configuré : {model_type.upper()} sur {self.device}")
        
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def prepare_data(self, csv_path: str, test_size: float = 0.2, 
                     batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """Prépare les DataLoaders d'entraînement et validation"""
        
        # Chargement des données
        df = pd.read_csv(csv_path)
        print(f"Données chargées: {len(df)} échantillons depuis {csv_path}")
        
        # Création du dataset
        use_sequences = (self.model_type == "lstm")
        dataset = TradingDataset(df, use_sequences=use_sequences)
        
        # Division train/validation
        val_size = int(len(dataset) * test_size)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)}")
        
        # Initialisation du modèle
        input_dim = dataset.X.shape[1]  # Nombre de features
        
        if self.model_type == "lstm":
            self.model = LSTMSignalGenerator(input_dim=input_dim)
        else:
            self.model = SimpleMLP(input_dim=input_dim)
        
        self.model.to(self.device)
        
        # Sauvegarde du scaler et encoder pour plus tard
        self.scaler = dataset.scaler
        self.label_encoder = dataset.label_encoder
        
        return train_loader, val_loader
    