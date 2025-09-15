import torch
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass
import yfinance as yf
import pandas as pd
from pathlib import Path

from .models import LSTMSignalGenerator, SimpleMLP
from .technical_indicators import TechnicalIndicators

@dataclass
class TradingSignal:
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    timestamp: str
    reasoning: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None

class SignalGenerator:
    """
    🎯 Générateur de signaux en production
    
    Se contente de charger un modèle entraîné et de l'utiliser
    Séparation claire avec la phase d'entraînement
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.7):
        """
        Args:
            model_path: Chemin vers le modèle entraîné (.pth)
            confidence_threshold: Seuil minimum de confiance
        """
        
        self.confidence_threshold = confidence_threshold
        self.tech_indicators = TechnicalIndicators()
        
        # Chargement du modèle entraîné
        print(f"📁 Chargement du modèle: {model_path}")
        self._load_model(model_path)
        
        print(f"✅ Générateur initialisé (seuil: {confidence_threshold:.0%})")
    
    def _load_model(self, model_path: str):
        """Charge le modèle et ses composants"""
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Reconstruction du modèle selon son type
        model_type = checkpoint['model_type']
        
        if model_type == "lstm":
            self.model = LSTMSignalGenerator(input_dim=4)  # À adapter selon vos features
        else:
            self.model = SimpleMLP(input_dim=4)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Chargement des preprocesseurs
        self.scaler = checkpoint['scaler']
        self.label_encoder = checkpoint['label_encoder']
        
        print(f"🤖 Modèle {model_type.upper()} chargé avec succès")
    
    def generate_signal(self, symbol: str, market_data: Dict) -> TradingSignal:
        """
        Génère un signal de trading pour un symbole donné
        
        Args:
            symbol: Ticker (ex: "AAPL")
            market_data: Données de marché actuelles
            
        Returns:
            Signal de trading complet
        """
        
        # Extraction des features (version simplifiée pour l'exemple)
        features = self._extract_features(symbol, market_data)
        
        if features is None:
            return self._create_hold_signal(symbol, market_data, "Données insuffisantes")
        
        # Prédiction
        action, confidence = self._predict(features)
        
        # Filtrage par confiance
        if confidence < self.confidence_threshold:
            return self._create_hold_signal(symbol, market_data, 
                                          f"Confiance trop faible ({confidence:.1%})")
        
        # Calcul des niveaux de trading
        current_price = market_data.get('price', 0.0)
        stop_loss, take_profit = self._calculate_levels(action, current_price)
        
        # Raisonnement
        reasoning = f"IA: {action} ({confidence:.0%}) | Prix: ${current_price:.2f}"
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            price=current_price,
            timestamp=datetime.now().isoformat(),
            reasoning=reasoning,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self._calculate_position_size(action, current_price, stop_loss)
        )
    
    def _extract_features(self, symbol: str, market_data: Dict) -> Optional[np.ndarray]:
        """Extrait les features pour le modèle (version simplifiée)"""
        
        try:
            # Features basiques (à adapter selon votre modèle)
            sentiment = market_data.get('sentiment_score', 0.0)
            price = market_data.get('price', 0.0)
            
            # TODO: Ajouter les indicateurs techniques complets
            features = np.array([sentiment, price, 0.0, 0.0])  # Placeholder
            
            # Normalisation
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            return features_scaled[0]
            
        except Exception as e:
            print(f"⚠️ Erreur extraction features {symbol}: {e}")
            return None
    
    def _predict(self, features: np.ndarray) -> tuple[str, float]:
        """Fait la prédiction avec le modèle"""
        
        with torch.no_grad():
            # Conversion en tensor
            if len(features.shape) == 1:
                # Pour LSTM, créer une séquence
                features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
            else:
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Prédiction
            logits, confidence = self.model(features_tensor)
            
            # Conversion en action
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence_score = float(confidence.item())
            
            action = self.label_encoder.inverse_transform([predicted_class])[0]
            
            return action, confidence_score
    
    def _calculate_levels(self, action: str, price: float) -> tuple[Optional[float], Optional[float]]:
        """Calcule stop-loss et take-profit"""
        
        if action == "BUY":
            return price * 0.98, price * 1.06  # -2%, +6%
        elif action == "SELL":
            return price * 1.02, price * 0.94  # +2%, -6%
        else:
            return None, None
    
    def _calculate_position_size(self, action: str, price: float, 
                               stop_loss: Optional[float]) -> float:
        """Calcule la taille de position (simplifié)"""
        
        if action == "HOLD" or not stop_loss:
            return 0.0
        
        # Logique simple : risque 2% sur 10k
        risk_amount = 10000 * 0.02  # 200$
        price_risk = abs(price - stop_loss)
        
        if price_risk > 0:
            return risk_amount / price_risk
        else:
            return 0.0
    
    def _create_hold_signal(self, symbol: str, market_data: Dict, reason: str) -> TradingSignal:
        """Crée un signal HOLD"""
        
        return TradingSignal(
            symbol=symbol,
            action="HOLD",
            confidence=0.0,
            price=market_data.get('price', 0.0),
            timestamp=datetime.now().isoformat(),
            reasoning=reason
        )