import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.logger import setup_logger
except ImportError:
    # Fallback si le logger n'est pas disponible
    import logging
    def setup_logger(name):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

logger = setup_logger(__name__)

class TradingSignalModel(nn.Module):
    """
    Modèle hybride pour prédire les signaux de trading (BUY/SELL/HOLD)
    Combine les embeddings FinBERT avec les features numériques
    """

    def __init__(self, bert_model_name="ProsusAI/finbert", num_numerical_features=2,
                 num_classes=3, hidden_dim=256, dropout_rate=0.3):
        super(TradingSignalModel, self).__init__()

        self.bert_model_name = bert_model_name
        self.num_classes = num_classes

        # Charger FinBERT
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)

        # Figer une partie des couches BERT pour éviter le catastrophic forgetting
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:8]:  # Figer les 8 premières couches
            for param in layer.parameters():
                param.requires_grad = False

        # Dimensions
        bert_dim = self.bert.config.hidden_size  # 768 pour FinBERT

        # Couches pour les features numériques
        self.numerical_processor = nn.Sequential(
            nn.Linear(num_numerical_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Couche de fusion
        fusion_dim = bert_dim + 128  # BERT embeddings + processed numerical features

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        """Initialiser les poids des nouvelles couches"""
        for module in [self.numerical_processor, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask, numerical_features):
        """
        Forward pass
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask
            numerical_features: Tensor with [sentiment_score, price_variation]
        """
        # Obtenir les embeddings BERT
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = bert_outputs.last_hidden_state.mean(dim=1)  # Mean pooling

        # Traiter les features numériques
        numerical_processed = self.numerical_processor(numerical_features)

        # Fusionner les embeddings
        fused_features = torch.cat([bert_embeddings, numerical_processed], dim=1)

        # Classification
        logits = self.classifier(fused_features)

        return logits

    def encode_text(self, texts, max_length=512):
        """
        Encoder une liste de textes
        """
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return encoding

    def predict(self, texts, numerical_features, device='cpu'):
        """
        Prédire les signaux de trading
        """
        self.eval()
        self.to(device)

        with torch.no_grad():
            # Encoder le texte
            encoding = self.encode_text(texts)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            # Convertir les features numériques
            if isinstance(numerical_features, np.ndarray):
                numerical_features = torch.tensor(numerical_features, dtype=torch.float32)
            numerical_features = numerical_features.to(device)

            # Prédiction
            logits = self.forward(input_ids, attention_mask, numerical_features)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            return predictions.cpu().numpy(), probabilities.cpu().numpy()


class TradingDataset(torch.utils.data.Dataset):
    """
    Dataset personnalisé pour les données de trading
    """

    def __init__(self, df, tokenizer, max_length=512, label_encoder=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Encoder les labels
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(df['action'])
        else:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(df['action'])

        # Préparer les features numériques
        self.numerical_features = self._prepare_numerical_features()

        logger.info(f"Dataset créé avec {len(self.df)} échantillons")
        logger.info(f"Classes: {self.label_encoder.classes_}")

    def _prepare_numerical_features(self):
        """
        Préparer les features numériques
        """
        features = []
        for _, row in self.df.iterrows():
            # Feature 1: Score de sentiment
            sentiment = row.get('sentiment_score', 0.0)

            # Feature 2: Variation de prix (si disponible)
            if 'variation' in row:
                variation = row['variation']
            elif 'price_now' in row and 'price_future' in row:
                variation = (row['price_future'] - row['price_now']) / row['price_now']
            else:
                variation = 0.0

            features.append([sentiment, variation])

        return np.array(features, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Obtenir le texte
        text = row['text']
        if pd.isna(text):
            text = ""

        # Tokenizer le texte
        encoding = self.tokenizer(
            str(text),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'numerical_features': torch.tensor(self.numerical_features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def create_data_loaders(df, tokenizer, train_ratio=0.8, batch_size=16, max_length=512):
    """
    Créer les data loaders pour l'entraînement et la validation
    """
    # Split train/validation
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    # Créer les datasets
    train_dataset = TradingDataset(train_df, tokenizer, max_length)
    val_dataset = TradingDataset(val_df, tokenizer, max_length, train_dataset.label_encoder)

    # Créer les data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Pour éviter les problèmes multiprocessing
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, train_dataset.label_encoder


if __name__ == "__main__":
    # Test du modèle
    model = TradingSignalModel()
    print(f"Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
    print(f"Paramètres entraînables: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")