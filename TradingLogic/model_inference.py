import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from trading_model import TradingSignalModel
from utils.logger import setup_logger

logger = setup_logger(__name__)

class TradingModelInference:
    """
    Classe pour l'inférence et l'évaluation du modèle de trading
    """

    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = None
        self.label_encoder = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """
        Charger un modèle pré-entraîné
        """
        model_path = Path(model_path)

        if model_path.is_dir():
            # Si c'est un dossier, chercher le fichier .pth
            model_file = model_path / 'trading_model.pth'
        else:
            model_file = model_path

        if not model_file.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_file}")

        logger.info(f"Chargement du modèle depuis: {model_file}")

        # Charger les données sauvegardées
        checkpoint = torch.load(model_file, map_location=self.device)

        self.config = checkpoint['config']
        self.label_encoder = checkpoint['label_encoder']

        # Recréer le modèle
        num_classes = len(self.label_encoder.classes_)
        self.model = TradingSignalModel(
            bert_model_name=self.config['bert_model_name'],
            num_classes=num_classes,
            hidden_dim=self.config.get('hidden_dim', 256),
            dropout_rate=self.config.get('dropout_rate', 0.3)
        )

        # Charger les poids
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Modèle chargé avec succès. Classes: {self.label_encoder.classes_}")

    def predict(self, texts, sentiment_scores=None, price_variations=None):
        """
        Prédire les signaux de trading pour une liste de textes

        Args:
            texts: Liste de textes d'actualités
            sentiment_scores: Liste de scores de sentiment (optionnel)
            price_variations: Liste de variations de prix (optionnel)

        Returns:
            predictions: Prédictions de classe
            probabilities: Probabilités pour chaque classe
            class_names: Noms des classes
        """
        if self.model is None:
            raise ValueError("Aucun modèle chargé. Utilisez load_model() d'abord.")

        # Préparer les features numériques
        if sentiment_scores is None:
            sentiment_scores = [0.0] * len(texts)
        if price_variations is None:
            price_variations = [0.0] * len(texts)

        numerical_features = np.array([
            [sentiment, variation]
            for sentiment, variation in zip(sentiment_scores, price_variations)
        ], dtype=np.float32)

        # Prédire
        predictions, probabilities = self.model.predict(
            texts,
            numerical_features,
            device=self.device
        )

        # Convertir les prédictions en noms de classe
        predicted_classes = self.label_encoder.inverse_transform(predictions)

        return predicted_classes, probabilities, self.label_encoder.classes_

    def predict_single(self, text, sentiment_score=0.0, price_variation=0.0):
        """
        Prédire le signal pour un seul texte
        """
        predictions, probabilities, classes = self.predict(
            [text],
            [sentiment_score],
            [price_variation]
        )

        result = {
            'prediction': predictions[0],
            'confidence': max(probabilities[0]),
            'probabilities': {
                class_name: prob
                for class_name, prob in zip(classes, probabilities[0])
            }
        }

        return result

    def evaluate_on_dataset(self, csv_path, save_results=True):
        """
        Évaluer le modèle sur un dataset complet
        """
        logger.info(f"Évaluation sur le dataset: {csv_path}")

        # Charger les données
        df = pd.read_csv(csv_path)

        # Préparer les inputs
        texts = df['text'].tolist()
        true_labels = df['action'].tolist()

        sentiment_scores = df.get('sentiment_score', [0.0] * len(df)).tolist()

        # Calculer les variations si disponible
        if 'variation' in df.columns:
            price_variations = df['variation'].tolist()
        elif 'price_now' in df.columns and 'price_future' in df.columns:
            price_variations = [
                (future - now) / now if now != 0 else 0.0
                for now, future in zip(df['price_now'], df['price_future'])
            ]
        else:
            price_variations = [0.0] * len(df)

        # Prédire
        predictions, probabilities, classes = self.predict(
            texts, sentiment_scores, price_variations
        )

        # Calculer les métriques
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("\nRapport de classification:")
        print(classification_report(true_labels, predictions))

        # Matrice de confusion
        cm = confusion_matrix(true_labels, predictions, labels=classes)

        if save_results:
            self._plot_confusion_matrix(cm, classes)
            self._save_evaluation_results(
                true_labels, predictions, probabilities, classes, accuracy
            )

        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities,
            'confusion_matrix': cm
        }

    def _plot_confusion_matrix(self, cm, classes, save_path=None):
        """
        Tracer la matrice de confusion
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes
        )
        plt.title('Matrice de Confusion')
        plt.xlabel('Prédictions')
        plt.ylabel('Vraies Étiquettes')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Matrice de confusion sauvegardée: {save_path}")

        plt.show()

    def _save_evaluation_results(self, true_labels, predictions, probabilities, classes, accuracy):
        """
        Sauvegarder les résultats d'évaluation
        """
        results = {
            'accuracy': accuracy,
            'classification_report': classification_report(
                true_labels, predictions, target_names=classes, output_dict=True
            ),
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_config': self.config
        }

        # Sauvegarder en JSON
        results_path = Path('evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Résultats sauvegardés: {results_path}")

    def analyze_predictions(self, texts, true_labels, predictions, probabilities,
                          save_errors=True, top_k=10):
        """
        Analyser les prédictions et identifier les erreurs
        """
        df_results = pd.DataFrame({
            'text': texts,
            'true_label': true_labels,
            'predicted_label': predictions,
            'confidence': [max(prob) for prob in probabilities],
            'correct': np.array(predictions) == np.array(true_labels)
        })

        # Statistiques générales
        accuracy = df_results['correct'].mean()
        avg_confidence = df_results['confidence'].mean()

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Confiance moyenne: {avg_confidence:.4f}")

        # Erreurs les plus confiantes (faux positifs)
        errors = df_results[~df_results['correct']].sort_values('confidence', ascending=False)

        logger.info(f"\nTop {top_k} erreurs les plus confiantes:")
        for i, row in errors.head(top_k).iterrows():
            logger.info(f"Vrai: {row['true_label']}, Prédit: {row['predicted_label']}, "
                       f"Confiance: {row['confidence']:.3f}")
            logger.info(f"Texte: {row['text'][:100]}...")
            logger.info("-" * 80)

        # Prédictions correctes les moins confiantes
        correct_low_conf = df_results[df_results['correct']].sort_values('confidence')

        logger.info(f"\nTop {top_k} prédictions correctes les moins confiantes:")
        for i, row in correct_low_conf.head(top_k).iterrows():
            logger.info(f"Label: {row['true_label']}, Confiance: {row['confidence']:.3f}")
            logger.info(f"Texte: {row['text'][:100]}...")
            logger.info("-" * 80)

        if save_errors:
            errors_path = Path('prediction_errors.csv')
            errors.to_csv(errors_path, index=False)
            logger.info(f"Erreurs sauvegardées: {errors_path}")

        return df_results

    def batch_predict_from_csv(self, input_csv, output_csv):
        """
        Faire des prédictions sur un fichier CSV et sauvegarder les résultats
        """
        logger.info(f"Prédiction par batch depuis: {input_csv}")

        df = pd.read_csv(input_csv)

        # Préparer les données
        texts = df['text'].tolist()
        sentiment_scores = df.get('sentiment_score', [0.0] * len(df)).tolist()

        if 'variation' in df.columns:
            price_variations = df['variation'].tolist()
        else:
            price_variations = [0.0] * len(df)

        # Prédire
        predictions, probabilities, classes = self.predict(
            texts, sentiment_scores, price_variations
        )

        # Ajouter les résultats au DataFrame
        df['predicted_action'] = predictions
        df['prediction_confidence'] = [max(prob) for prob in probabilities]

        # Ajouter les probabilités pour chaque classe
        for i, class_name in enumerate(classes):
            df[f'prob_{class_name}'] = [prob[i] for prob in probabilities]

        # Sauvegarder
        df.to_csv(output_csv, index=False)
        logger.info(f"Prédictions sauvegardées: {output_csv}")

        return df


def main():
    """
    Exemple d'utilisation du modèle d'inférence
    """
    # Chemins
    base_dir = Path(__file__).resolve().parent.parent

    # Trouver le dernier modèle entraîné
    models_dir = base_dir / "models"
    if models_dir.exists():
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('trading_model_')]
        if model_dirs:
            latest_model = max(model_dirs, key=lambda x: x.name)
            logger.info(f"Utilisation du modèle: {latest_model}")

            # Créer l'interface d'inférence
            inference = TradingModelInference(latest_model)

            # Test sur quelques exemples
            test_texts = [
                "Apple reports strong quarterly earnings with revenue up 15%",
                "Market crash expected as inflation soars to new heights",
                "Tesla announces new factory opening in Europe"
            ]

            for text in test_texts:
                result = inference.predict_single(text)
                logger.info(f"Texte: {text}")
                logger.info(f"Prédiction: {result['prediction']} (confiance: {result['confidence']:.3f})")
                logger.info(f"Probabilités: {result['probabilities']}")
                logger.info("-" * 80)

        else:
            logger.error("Aucun modèle trouvé dans le dossier models/")
    else:
        logger.error("Dossier models/ non trouvé")


if __name__ == "__main__":
    main()