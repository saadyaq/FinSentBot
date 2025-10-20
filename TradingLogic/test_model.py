#!/usr/bin/env python3
"""
Script de test rapide pour le modèle de trading
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from TradingLogic.model_inference import TradingModelInference
from utils.logger import setup_logger

logger = setup_logger(__name__)

def test_model_loading():
    """
    Tester le chargement du modèle
    """
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"

    if not models_dir.exists():
        logger.error("Dossier models/ non trouvé. Entraînez d'abord un modèle.")
        return None

    # Trouver le dernier modèle
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('trading_model_')]
    if not model_dirs:
        logger.error("Aucun modèle trouvé dans models/")
        return None

    latest_model = max(model_dirs, key=lambda x: x.name)
    logger.info(f"Test du modèle: {latest_model.name}")

    try:
        inference = TradingModelInference(latest_model)
        logger.info("✅ Modèle chargé avec succès")
        return inference
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement: {e}")
        return None

def test_predictions(inference):
    """
    Tester les prédictions sur quelques exemples
    """
    if inference is None:
        return

    logger.info("=== Test des prédictions ===")

    # Exemples de test avec différents sentiments
    test_cases = [
        {
            'text': "Apple reports record quarterly earnings beating all analyst expectations with revenue up 25%",
            'sentiment': 0.8,
            'variation': 0.05,
            'expected': 'BUY'
        },
        {
            'text': "Market crash imminent as inflation soars and unemployment rises to decade highs",
            'sentiment': -0.9,
            'variation': -0.08,
            'expected': 'SELL'
        },
        {
            'text': "Tesla stock remains stable after mixed quarterly results with slight revenue increase",
            'sentiment': 0.1,
            'variation': 0.002,
            'expected': 'HOLD'
        },
        {
            'text': "Company announces massive layoffs and plant closures amid declining demand",
            'sentiment': -0.7,
            'variation': -0.12,
            'expected': 'SELL'
        },
        {
            'text': "New breakthrough technology could revolutionize the industry and boost profits",
            'sentiment': 0.9,
            'variation': 0.15,
            'expected': 'BUY'
        }
    ]

    correct_predictions = 0
    total_predictions = len(test_cases)

    for i, case in enumerate(test_cases, 1):
        logger.info(f"\n--- Test {i}/{total_predictions} ---")
        logger.info(f"Texte: {case['text'][:80]}...")
        logger.info(f"Sentiment: {case['sentiment']}, Variation: {case['variation']}")
        logger.info(f"Attendu: {case['expected']}")

        try:
            result = inference.predict_single(
                case['text'],
                case['sentiment'],
                case['variation']
            )

            prediction = result['prediction']
            confidence = result['confidence']
            probabilities = result['probabilities']

            logger.info(f"Prédit: {prediction} (confiance: {confidence:.3f})")
            logger.info(f"Probabilités: {probabilities}")

            if prediction == case['expected']:
                logger.info("✅ Prédiction correcte")
                correct_predictions += 1
            else:
                logger.info("❌ Prédiction incorrecte")

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")

    # Résumé
    accuracy = correct_predictions / total_predictions
    logger.info(f"\n=== Résumé des tests ===")
    logger.info(f"Prédictions correctes: {correct_predictions}/{total_predictions}")
    logger.info(f"Accuracy sur les tests: {accuracy:.2%}")

    if accuracy >= 0.8:
        logger.info("🎉 Excellent! Le modèle fonctionne bien.")
    elif accuracy >= 0.6:
        logger.info("👍 Bien. Le modèle a des performances acceptables.")
    else:
        logger.info("⚠️ Attention. Le modèle pourrait nécessiter plus d'entraînement.")

def test_batch_prediction():
    """
    Tester la prédiction par batch
    """
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "training_datasets" / "train_enhanced_with_historical.csv"

    if not data_path.exists():
        logger.warning("Dataset non trouvé pour le test batch")
        return

    models_dir = base_dir / "models"
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('trading_model_')]

    if not model_dirs:
        return

    latest_model = max(model_dirs, key=lambda x: x.name)

    try:
        inference = TradingModelInference(latest_model)

        logger.info("=== Test d'évaluation sur dataset ===")
        results = inference.evaluate_on_dataset(data_path, save_results=False)

        logger.info(f"Accuracy sur le dataset complet: {results['accuracy']:.4f}")

    except Exception as e:
        logger.error(f"Erreur lors du test batch: {e}")

def main():
    """
    Fonction principale de test
    """
    logger.info("🧪 Test du modèle de trading")

    # Test 1: Chargement du modèle
    inference = test_model_loading()

    # Test 2: Prédictions individuelles
    if inference:
        test_predictions(inference)

        # Test 3: Évaluation sur dataset (optionnel)
        logger.info("\n" + "="*60)
        test_batch_prediction()

    logger.info("\n🏁 Tests terminés")

if __name__ == "__main__":
    main()