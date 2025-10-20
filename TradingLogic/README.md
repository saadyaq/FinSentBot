# Modèle de Trading avec FinBERT

Ce module implémente un modèle hybride pour prédire les signaux de trading (BUY/SELL/HOLD) basé sur l'analyse de sentiment financier et les données de prix historiques.

## Architecture du Modèle

Le modèle combine :
- **FinBERT** : Modèle BERT pré-entraîné sur des données financières pour l'analyse de sentiment
- **Features numériques** : Scores de sentiment et variations de prix
- **Réseau de neurones** : Couches fully-connected pour la fusion et la classification

### Caractéristiques
- Classes de sortie : BUY, SELL, HOLD
- Gestion du déséquilibre des classes avec des poids automatiques
- Early stopping pour éviter le surapprentissage
- Visualisation de l'entraînement et métriques d'évaluation

## Installation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

2. Vérifier que PyTorch est installé avec le support GPU (optionnel) :
```bash
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"
```

## Dataset

Le modèle s'entraîne sur `train_enhanced_with_historical.csv` qui contient :
- **symbol** : Symbole de l'action (ex: AAPL, TSLA)
- **text** : Texte de l'actualité financière
- **sentiment_score** : Score de sentiment (-1 à 1)
- **price_now** : Prix au moment de la news
- **price_future** : Prix après la fenêtre d'observation
- **variation** : Variation de prix calculée
- **action** : Label de trading (BUY/SELL/HOLD)

## Utilisation

### 1. Entraînement du modèle

#### Méthode rapide :
```bash
cd TradingLogic
python run_training.py
```

#### Avec paramètres personnalisés :
```bash
python run_training.py \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --epochs 10 \
    --max-length 512 \
    --sample-size 1000  # Pour test rapide
```

#### Paramètres disponibles :
- `--data-path` : Chemin vers le dataset CSV
- `--output-dir` : Dossier de sortie pour le modèle
- `--batch-size` : Taille de batch (défaut: 8)
- `--learning-rate` : Taux d'apprentissage (défaut: 1e-5)
- `--epochs` : Nombre d'époques (défaut: 5)
- `--max-length` : Longueur max des séquences (défaut: 256)
- `--hidden-dim` : Dimension couches cachées (défaut: 128)
- `--dropout` : Taux de dropout (défaut: 0.3)
- `--patience` : Patience early stopping (défaut: 2)
- `--train-ratio` : Ratio train/validation (défaut: 0.8)
- `--sample-size` : Échantillonnage pour test (optionnel)

### 2. Test du modèle

```bash
python test_model.py
```

Ce script :
- Charge le dernier modèle entraîné
- Teste sur des exemples prédéfinis
- Évalue les performances
- Affiche l'accuracy

### 3. Utilisation programmatique

#### Charger et utiliser un modèle :

```python
from model_inference import TradingModelInference

# Charger le modèle
inference = TradingModelInference('models/trading_model_20241020_143052')

# Prédiction simple
result = inference.predict_single(
    text="Apple reports strong quarterly earnings",
    sentiment_score=0.8,
    price_variation=0.05
)

print(f"Prédiction: {result['prediction']}")
print(f"Confiance: {result['confidence']:.3f}")
print(f"Probabilités: {result['probabilities']}")
```

#### Prédiction par batch :

```python
# Prédire sur plusieurs textes
texts = [
    "Company beats earnings expectations",
    "Market crash expected due to inflation",
    "Stable performance with modest growth"
]
sentiment_scores = [0.7, -0.8, 0.1]
price_variations = [0.08, -0.12, 0.02]

predictions, probabilities, classes = inference.predict(
    texts, sentiment_scores, price_variations
)

for i, pred in enumerate(predictions):
    print(f"Texte {i+1}: {pred} (confiance: {max(probabilities[i]):.3f})")
```

#### Évaluation sur dataset :

```python
# Évaluer sur un fichier CSV
results = inference.evaluate_on_dataset(
    'data/training_datasets/test_set.csv',
    save_results=True
)

print(f"Accuracy: {results['accuracy']:.4f}")
```

## Structure des fichiers

```
TradingLogic/
├── README.md                 # Ce fichier
├── trading_model.py          # Architecture du modèle
├── train_model.py           # Script d'entraînement
├── model_inference.py       # Interface d'inférence
├── run_training.py          # Script utilitaire d'entraînement
├── test_model.py           # Tests automatisés
└── prepare_dataset.py      # Préparation des données (existant)
```

## Configuration recommandée

### Pour un entraînement rapide (test) :
```bash
python run_training.py --sample-size 1000 --epochs 3 --batch-size 8
```

### Pour un entraînement complet :
```bash
python run_training.py --batch-size 16 --epochs 10 --max-length 512 --learning-rate 2e-5
```

### Avec GPU :
```bash
python run_training.py --gpu --batch-size 32 --epochs 15
```

## Métriques et Évaluation

Le modèle génère automatiquement :
- **Graphiques d'entraînement** : Loss et accuracy par époque
- **Matrice de confusion** : Visualisation des prédictions
- **Rapport de classification** : Précision, rappel, F1-score par classe
- **Analyse des erreurs** : Exemples de prédictions incorrectes

## Optimisation

### Améliorer les performances :
1. **Augmenter les données** : Plus d'échantillons d'entraînement
2. **Ajuster les hyperparamètres** : Learning rate, batch size, architecture
3. **Features engineering** : Ajouter plus de features numériques
4. **Équilibrage des classes** : Techniques de sur/sous-échantillonnage

### Réduire le temps d'entraînement :
1. **Réduire max_length** : 256 au lieu de 512
2. **Figer plus de couches BERT** : Modifier `trading_model.py`
3. **Batch size plus grand** : Si mémoire GPU suffisante
4. **Échantillonnage** : Utiliser `--sample-size` pour les tests

## Troubleshooting

### Erreur de mémoire GPU :
- Réduire `--batch-size` (essayer 4 ou 8)
- Réduire `--max-length` (essayer 128 ou 256)
- Fermer d'autres processus utilisant le GPU

### Modèle qui ne converge pas :
- Vérifier la qualité des données
- Ajuster le learning rate (essayer 5e-6 ou 1e-4)
- Augmenter le nombre d'époques
- Vérifier l'équilibrage des classes

### Performances médiocres :
- Vérifier la distribution des classes dans le dataset
- Analyser les erreurs avec `model_inference.py`
- Ajuster les seuils de classification dans `prepare_dataset.py`
- Ajouter plus de features numériques

## Intégration avec le système existant

Le modèle peut être intégré avec :
- **Scrapers de news** : Pour obtenir les textes en temps réel
- **API de prix** : Pour les données de marché
- **Système de trading** : Pour automatiser les décisions
- **Dashboard Streamlit** : Pour la visualisation

Exemple d'intégration :
```python
from nlp.sentiment_model import SentimentModel
from TradingLogic.model_inference import TradingModelInference

# Pipeline complet
sentiment_analyzer = SentimentModel()
trading_model = TradingModelInference('models/latest')

def analyze_news(news_text, current_price, future_price):
    # Analyser le sentiment
    sentiment = sentiment_analyzer.predict_sentiment(news_text)

    # Calculer la variation
    variation = (future_price - current_price) / current_price

    # Prédire l'action
    result = trading_model.predict_single(news_text, sentiment, variation)

    return result
```