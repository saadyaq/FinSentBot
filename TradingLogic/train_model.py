import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime

from trading_model import TradingSignalModel, create_data_loaders
from utils.logger import setup_logger

logger = setup_logger(__name__)

class TradingModelTrainer:
    """
    Entraîneur pour le modèle de signaux de trading
    """

    def __init__(self, model_config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Utilisation du device: {self.device}")

        # Configuration par défaut
        self.config = {
            'bert_model_name': 'ProsusAI/finbert',
            'max_length': 512,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'num_epochs': 10,
            'hidden_dim': 256,
            'dropout_rate': 0.3,
            'patience': 3,
            'train_ratio': 0.8,
            'weight_decay': 0.01
        }

        if model_config:
            self.config.update(model_config)

        self.model = None
        self.label_encoder = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def load_data(self, csv_path):
        """
        Charger les données d'entraînement
        """
        logger.info(f"Chargement des données depuis: {csv_path}")
        df = pd.read_csv(csv_path)

        # Vérifications de base
        required_columns = ['text', 'action']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes: {missing_columns}")

        # Nettoyer les données
        df = df.dropna(subset=required_columns)
        df = df[df['text'].str.len() > 10]  # Filtrer les textes trop courts

        logger.info(f"Données chargées: {len(df)} échantillons")
        logger.info(f"Distribution des classes: {df['action'].value_counts().to_dict()}")

        return df

    def create_model(self, num_classes):
        """
        Créer le modèle de trading
        """
        model = TradingSignalModel(
            bert_model_name=self.config['bert_model_name'],
            num_classes=num_classes,
            hidden_dim=self.config['hidden_dim'],
            dropout_rate=self.config['dropout_rate']
        )

        # Calculer les poids de classe pour gérer le déséquilibre
        return model

    def calculate_class_weights(self, labels):
        """
        Calculer les poids des classes pour le loss balancé
        """
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        return torch.tensor(class_weights, dtype=torch.float32, device=self.device)

    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """
        Entraîner le modèle pour une époque
        """
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, batch in enumerate(train_loader):
            # Déplacer les données vers le device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            numerical_features = batch['numerical_features'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, numerical_features)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Statistiques
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            if batch_idx % 50 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                           f'Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def validate_epoch(self, model, val_loader, criterion):
        """
        Valider le modèle
        """
        model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids, attention_mask, numerical_features)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def train(self, df):
        """
        Entraîner le modèle complet
        """
        logger.info("Début de l'entraînement")

        # Créer le tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config['bert_model_name'])

        # Créer les data loaders
        train_loader, val_loader, label_encoder = create_data_loaders(
            df,
            tokenizer=tokenizer,
            train_ratio=self.config['train_ratio'],
            batch_size=self.config['batch_size'],
            max_length=self.config['max_length']
        )

        self.label_encoder = label_encoder
        num_classes = len(label_encoder.classes_)

        # Créer le modèle
        self.model = self.create_model(num_classes)
        self.model.to(self.device)

        logger.info(f"Modèle créé avec {sum(p.numel() for p in self.model.parameters())} paramètres")
        logger.info(f"Paramètres entraînables: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        # Calculer les poids de classe
        train_labels = []
        for batch in train_loader:
            train_labels.extend(batch['labels'].numpy())
        class_weights = self.calculate_class_weights(train_labels)

        # Optimiseur et critère
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        # Variables pour early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Boucle d'entraînement
        for epoch in range(self.config['num_epochs']):
            start_time = time.time()

            # Entraînement
            train_loss, train_acc = self.train_epoch(self.model, train_loader, optimizer, criterion, epoch + 1)

            # Validation
            val_loss, val_acc = self.validate_epoch(self.model, val_loader, criterion)

            # Scheduler
            scheduler.step(val_loss)

            # Historique
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)

            epoch_time = time.time() - start_time

            logger.info(f'Epoch {epoch + 1}/{self.config["num_epochs"]} - '
                       f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - '
                       f'Time: {epoch_time:.2f}s')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                logger.info(f"Nouveau meilleur modèle sauvegardé (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= self.config['patience']:
                logger.info(f"Early stopping à l'époque {epoch + 1}")
                break

        # Charger le meilleur modèle
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info("Meilleur modèle chargé")

        return self.model

    def evaluate_model(self, val_loader):
        """
        Évaluer le modèle sur l'ensemble de validation
        """
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask, numerical_features)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Métriques
        accuracy = accuracy_score(all_labels, all_predictions)
        class_names = self.label_encoder.classes_

        logger.info(f"Accuracy finale: {accuracy:.4f}")
        logger.info("Rapport de classification:")
        logger.info(classification_report(all_labels, all_predictions, target_names=class_names))

        return all_predictions, all_labels

    def save_model(self, save_dir):
        """
        Sauvegarder le modèle et ses métadonnées
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le modèle
        model_path = save_dir / 'trading_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'label_encoder': self.label_encoder,
            'training_history': self.training_history
        }, model_path)

        # Sauvegarder la configuration
        config_path = save_dir / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        logger.info(f"Modèle sauvegardé dans: {save_dir}")

    def plot_training_history(self, save_dir=None):
        """
        Tracer l'historique d'entraînement
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        ax1.plot(self.training_history['train_loss'], label='Train Loss')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Accuracy
        ax2.plot(self.training_history['train_acc'], label='Train Accuracy')
        ax2.plot(self.training_history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()

        if save_dir:
            plot_path = Path(save_dir) / 'training_history.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique sauvegardé: {plot_path}")

        plt.show()


def main():
    """
    Fonction principale d'entraînement
    """
    # Chemins
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "training_datasets" / "train_enhanced_with_historical.csv"
    model_save_dir = base_dir / "models" / f"trading_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Vérifier que le fichier existe
    if not data_path.exists():
        logger.error(f"Fichier de données non trouvé: {data_path}")
        return

    # Configuration du modèle
    config = {
        'batch_size': 8,  # Réduit pour éviter les problèmes de mémoire
        'learning_rate': 1e-5,
        'num_epochs': 5,
        'max_length': 256,  # Réduit pour la rapidité
        'hidden_dim': 128,
        'patience': 2
    }

    # Créer l'entraîneur
    trainer = TradingModelTrainer(config)

    try:
        # Charger les données
        df = trainer.load_data(data_path)

        # Limiter pour test rapide (optionnel)
        # df = df.sample(1000).reset_index(drop=True)  # Décommenter pour test rapide

        # Entraîner le modèle
        model = trainer.train(df)

        # Créer les data loaders pour l'évaluation
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.get('bert_model_name', 'ProsusAI/finbert'))
        _, val_loader, _ = create_data_loaders(
            df,
            tokenizer=tokenizer,
            train_ratio=config['train_ratio'],
            batch_size=config['batch_size'],
            max_length=config['max_length']
        )

        # Évaluer le modèle
        predictions, labels = trainer.evaluate_model(val_loader)

        # Sauvegarder le modèle
        trainer.save_model(model_save_dir)

        # Tracer l'historique
        trainer.plot_training_history(model_save_dir)

        logger.info("Entraînement terminé avec succès!")

    except Exception as e:
        logger.error(f"Erreur durant l'entraînement: {e}")
        raise


if __name__ == "__main__":
    main()