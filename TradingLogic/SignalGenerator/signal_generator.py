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
        self.feature_cols: list[str] = []
        self.sequence_length: int = 1
        self.model_type: str = "mlp"
        
        # Chargement du modèle entraîné
        print(f"📁 Chargement du modèle: {model_path}")
        self._load_model(model_path)
        
        print(f"✅ Générateur initialisé (seuil: {confidence_threshold:.0%})")
    
    def _load_model(self, model_path: str):
        """Charge le modèle et ses composants"""
        
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        self.model_type = checkpoint.get("model_type", "lstm")
        self.feature_cols = checkpoint.get(
            "feature_cols",
            ["sentiment_score", "price_now", "price_future", "variation"],
        )
        self.sequence_length = int(checkpoint.get("sequence_length", 1))

        input_dim = len(self.feature_cols)
        if input_dim == 0:
            input_dim = 4

        if self.model_type == "lstm":
            self.model = LSTMSignalGenerator(input_dim=input_dim)
        else:
            self.model = SimpleMLP(input_dim=input_dim)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Chargement des preprocesseurs
        self.scaler = checkpoint['scaler']
        self.label_encoder = checkpoint['label_encoder']
        
        print(f"🤖 Modèle {self.model_type.upper()} chargé avec succès")
    
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
            values = []
            for col in self.feature_cols:
                if col == "price_now":
                    values.append(float(market_data.get("price", 0.0)))
                elif col == "price_future":
                    values.append(float(market_data.get("price_future", market_data.get("price", 0.0))))
                elif col == "variation":
                    if "variation" in market_data:
                        values.append(float(market_data["variation"]))
                    elif "price" in market_data and "previous_price" in market_data:
                        prev_price = market_data["previous_price"]
                        current_price = market_data["price"]
                        if prev_price:
                            values.append(float(current_price - prev_price) / float(prev_price))
                        else:
                            values.append(0.0)
                    else:
                        values.append(0.0)
                else:
                    values.append(float(market_data.get(col, 0.0)))
            
            features = np.array(values, dtype=np.float32)
            
            # Normalisation
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            return features_scaled[0]
            
        except Exception as e:
            print(f"⚠️ Erreur extraction features {symbol}: {e}")
            return None
    
    def _predict(self, features: np.ndarray) -> tuple[str, float]:
        """Fait la prédiction avec le modèle"""
        
        with torch.no_grad():
            features_array = np.asarray(features, dtype=np.float32)

            if self.model_type == "lstm":
                if features_array.ndim == 1:
                    sequence = np.tile(features_array, (self.sequence_length, 1))
                else:
                    sequence = features_array.astype(np.float32)
                    if sequence.shape[0] < self.sequence_length:
                        pad_len = self.sequence_length - sequence.shape[0]
                        padding = np.zeros(
                            (pad_len, sequence.shape[1]), dtype=np.float32
                        )
                        sequence = np.vstack((padding, sequence))
                    elif sequence.shape[0] > self.sequence_length:
                        sequence = sequence[-self.sequence_length :]
                features_tensor = torch.from_numpy(sequence).unsqueeze(0)
            else:
                if features_array.ndim > 1:
                    features_array = features_array[-1]
                features_tensor = torch.from_numpy(features_array).unsqueeze(0)
            
            # Prédiction
            logits, confidence = self.model(features_tensor.float())
            
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
