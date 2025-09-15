import torch
import torch.nn as nn
from typing import Tuple

class LSTMSignalGenerator(nn.Module):
    """
    🧠 Modèle LSTM avec mécanisme d'attention pour signaux de trading
    
    Architecture :
    - LSTM multi-couches pour capturer les séquences temporelles
    - Attention pour se concentrer sur les patterns importants  
    - Double sortie : prédiction + confiance
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 num_layers: int = 2, output_dim: int = 3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Couches LSTM avec dropout pour éviter l'overfitting
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Mécanisme d'attention multi-têtes
        self.attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=4, 
            batch_first=True
        )
        
        # Classificateur pour l'action (SELL/HOLD/BUY)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Estimateur de confiance
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Sortie entre 0 et 1
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass du modèle
        
        Args:
            x: Séquence de features (batch_size, seq_length, input_dim)
            
        Returns:
            logits: Scores pour chaque classe (batch_size, 3)
            confidence: Score de confiance (batch_size, 1)
        """
        # Passage dans LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        
        # Application de l'attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Utilisation du dernier timestep
        final_hidden = attended[:, -1, :]  # (batch, hidden)
        
        # Prédictions
        logits = self.classifier(final_hidden)
        confidence = self.confidence_estimator(final_hidden)
        
        return logits, confidence

class SimpleMLP(nn.Module):
    """
    🔨 Modèle MLP simple pour comparaison/prototypage rapide
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 3):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.confidence = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass pour MLP simple"""
        logits = self.network(x)
        conf = self.confidence(x)
        return logits, conf