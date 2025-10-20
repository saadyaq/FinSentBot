#!/usr/bin/env python3
"""
Script utilitaire pour lancer l'entraînement du modèle de trading
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from TradingLogic.train_model import TradingModelTrainer
from utils.logger import setup_logger

logger = setup_logger(__name__)

def parse_arguments():
    """
    Parser les arguments de ligne de commande
    """
    parser = argparse.ArgumentParser(description='Entraîner le modèle de signaux de trading')

    parser.add_argument(
        '--data-path',
        type=str,
        default='data/training_datasets/train_enhanced_with_historical.csv',
        help='Chemin vers le fichier de données CSV'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Répertoire de sortie pour sauvegarder le modèle'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Taille de batch pour l\'entraînement'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5,
        help='Taux d\'apprentissage'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Nombre d\'époques d\'entraînement'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=256,
        help='Longueur maximale des séquences de texte'
    )

    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=128,
        help='Dimension des couches cachées'
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Taux de dropout'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=2,
        help='Patience pour l\'early stopping'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Ratio des données pour l\'entraînement (le reste pour la validation)'
    )

    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Nombre d\'échantillons à utiliser (pour test rapide)'
    )

    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Utiliser le GPU si disponible'
    )

    return parser.parse_args()

def main():
    """
    Fonction principale
    """
    args = parse_arguments()

    # Chemins
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / args.data_path

    # Vérifier que le fichier existe
    if not data_path.exists():
        logger.error(f"Fichier de données non trouvé: {data_path}")
        sys.exit(1)

    # Configuration du modèle basée sur les arguments
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.epochs,
        'max_length': args.max_length,
        'hidden_dim': args.hidden_dim,
        'dropout_rate': args.dropout,
        'patience': args.patience,
        'train_ratio': args.train_ratio
    }

    # Répertoire de sortie
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        output_dir = base_dir / "models" / f"trading_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("=== Configuration d'entraînement ===")
    logger.info(f"Données: {data_path}")
    logger.info(f"Sortie: {output_dir}")
    logger.info(f"Configuration: {config}")

    try:
        # Créer l'entraîneur
        trainer = TradingModelTrainer(config)

        # Charger les données
        df = trainer.load_data(data_path)

        # Échantillonnage si demandé
        if args.sample_size and args.sample_size < len(df):
            df = df.sample(args.sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Échantillonnage: {len(df)} échantillons")

        # Entraîner le modèle
        logger.info("=== Début de l'entraînement ===")
        model = trainer.train(df)

        # Créer les data loaders pour l'évaluation finale
        from TradingLogic.trading_model import create_data_loaders
        _, val_loader, _ = create_data_loaders(
            df,
            tokenizer=None,
            train_ratio=config['train_ratio'],
            batch_size=config['batch_size'],
            max_length=config['max_length']
        )

        # Évaluation finale
        logger.info("=== Évaluation finale ===")
        predictions, labels = trainer.evaluate_model(val_loader)

        # Sauvegarder le modèle
        trainer.save_model(output_dir)

        # Tracer l'historique
        trainer.plot_training_history(output_dir)

        logger.info(f"=== Entraînement terminé avec succès! ===")
        logger.info(f"Modèle sauvegardé dans: {output_dir}")

    except KeyboardInterrupt:
        logger.info("Entraînement interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur durant l'entraînement: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()