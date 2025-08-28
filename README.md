# 🧠 FinSentBot

**FinSentBot** est un système de trading automatique avancé qui combine l'analyse de sentiment en temps réel des actualités financières avec la collecte de données boursières historiques et en temps réel. Le projet utilise **Apache Kafka** pour la gestion des flux de données, **FinBERT** pour l'analyse de sentiment financier, et un modèle de génération de signaux de trading basé sur l'apprentissage automatique.

---

## 📊 Fonctionnalités

### 🔄 Collecte de données en temps réel
- **Prix d'actions** via Yahoo Finance API
- **Scraping d'articles** depuis CNBC, CoinDesk, Financial Times, et TechCrunch
- **Données historiques** extensives avec multiple périodes et intervalles

### 🧠 Analyse intelligente
- **Analyse de sentiment** utilisant le modèle FinBERT (ProsusAI/finbert)
- **Génération de signaux de trading** basée sur ML
- **Preprocessing avancé** des données textuelles

### 📊 Visualisation et monitoring
- **Dashboard interactif** Streamlit
- **Interface Kafka UI** pour monitoring des topics
- **Logging complet** avec fichiers de logs rotatifs

---

## 🏗️ Architecture du projet

```bash
FinSentBot/
├── messaging/
│   ├── producers/
│   │   ├── news_scraper_producer.py
│   │   ├── stock_price_producer.py
│   │   └── scrapers/
│   │       └── src/
│   │           ├── scraper_cnbc.py
│   │           ├── scraper_coindesk.py
│   │           ├── scraper_ft.py
│   │           └── scraper_tc.py
│   └── consumers/
│       ├── sentiment_analysis_consumer.py
│       └── trading_signal_consumer.py
│
├── TradingLogic/
│   ├── SignalGenerator/
│   │   ├── model.py
│   │   └── train.py
│   ├── prepare_dataset.py
│   └── trained_model.pth
│
├── nlp/
│   ├── sentiment_model.py
│   ├── preprocessing.py
│   └── test.py
│
├── data/
│   ├── raw/
│   │   ├── news_sentiment.jsonl
│   │   ├── stock_prices.jsonl
│   │   └── seen_symbols.json
│   ├── logs/
│   ├── scripts/
│   │   ├── historical_stock_price_collector.py
│   │   ├── integrate_historical_prices.py
│   │   └── run_stock_expansion.py
│   └── training_datasets/
│       └── train.csv
│
├── dashboard/
│   └── app.py
│
├── config/
│   ├── kafka_config.py
│   └── settings.yaml
│
├── utils/
│   ├── helpers.py
│   └── logger.py
│
├── notebooks/
│   └── data_analytics.ipynb
│
├── requirements.txt
├── docker-compose.yaml
└── README.md

---

## 🚀 Démarrage rapide

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 2. Lancement de l'infrastructure Kafka

```bash
docker-compose up -d
```

Cela démarre :
- **Kafka** sur le port 9092
- **Zookeeper** sur le port 2181  
- **Kafka UI** sur le port 8080 (interface web)

### 3. Test de la configuration

```bash
python test_send.py
```

Ce script envoie un message test sur le topic `raw_news` pour vérifier la connectivité Kafka.

### 4. Collecte de données historiques

Pour enrichir votre dataset avec des données historiques S&P 500 :

```bash
python data/scripts/historical_stock_price_collector.py
```

Ce script collecte automatiquement les prix historiques pour 100+ symboles avec différentes périodes (1mo, 3mo, 6mo, 1y) et intervalles (1m, 5m, 30m, 1h).

### 5. Génération du dataset d'entraînement

```bash
python TradingLogic/prepare_dataset.py
```

Crée le fichier `data/training_datasets/train.csv` avec les échantillons alignés par symbole et timestamps.

### 6. Lancement des producteurs et consommateurs

**Producteur de news :**
```bash
python messaging/producers/news_scraper_producer.py
```

**Producteur de prix :**
```bash
python messaging/producers/stock_price_producer.py
```

**Consommateur d'analyse de sentiment :**
```bash
python messaging/consumers/sentiment_analysis_consumer.py
```

**Consommateur de signaux de trading :**
```bash
python messaging/consumers/trading_signal_consumer.py
```

### 7. Dashboard de visualisation

```bash
streamlit run dashboard/app.py
```

---

## 🔧 Technologies utilisées

- **Apache Kafka** - Streaming de données en temps réel
- **Python** - Langage principal  
- **FinBERT** (ProsusAI/finbert) - Modèle d'analyse de sentiment financier
- **PyTorch** - Framework de deep learning
- **Transformers** (HuggingFace) - Modèles NLP pré-entraînés
- **yfinance** - API Yahoo Finance pour les données boursières
- **Streamlit** - Dashboard interactif
- **BeautifulSoup** - Web scraping
- **Pandas** - Manipulation de données
- **Docker** - Containerisation de Kafka

---

## 📈 Flux de données

1. **Collecte** : Les scrapers récupèrent les articles depuis CNBC, CoinDesk, Financial Times, TechCrunch
2. **Streaming** : Les articles sont envoyés dans Kafka topic `raw_news`
3. **Analyse** : Le consommateur de sentiment analyse chaque article avec FinBERT
4. **Enrichissement** : Les prix d'actions correspondants sont collectés via Yahoo Finance
5. **Signaux** : Le générateur de signaux combine sentiment + prix pour produire Buy/Sell/Hold
6. **Visualisation** : Le dashboard affiche les résultats en temps réel

---

## 🗂️ Structure des données

### Articles de news (`news_sentiment.jsonl`)
```json
{
  "source": "CNBC",
  "title": "Market rallies on positive earnings",
  "content": "...",
  "url": "https://...",
  "timestamp": "2024-08-28T10:30:00",
  "sentiment_score": 0.7534
}
```

### Prix d'actions (`stock_prices.jsonl`)
```json
{
  "symbol": "AAPL",
  "timestamp": "2024-08-28T10:30:00",
  "price": 184.50,
  "open": 183.20,
  "high": 185.10,
  "low": 182.90,
  "volume": 1234567,
  "period": "1mo",
  "interval": "1h"
}
```

---

## 🎯 Utilisation avancée

### Collecte de données historiques personnalisée

Le collecteur historique supporte plusieurs modes de fonctionnement :

```python
from data.scripts.historical_stock_price_collector import HistoricalStockCollector

collector = HistoricalStockCollector()
# Collecte massive S&P 500
df = collector.run_massive_collection(save_frequency=5)
```

### Entraînement du modèle de signaux

```bash
python TradingLogic/SignalGenerator/train.py
```

### Monitoring avec Kafka UI

Accédez à http://localhost:8080 pour monitorer :
- Topics Kafka
- Messages en temps réel  
- Consommateurs actifs
- Métriques de performance

---

## 📊 Logging et debugging

Les logs sont automatiquement générés dans `data/logs/` avec rotation quotidienne :
- `finsentbot_YYYYMMDD.log` - Logs applicatifs
- Niveaux : INFO, WARNING, ERROR
- Format : timestamp - niveau - message

---

## ⚙️ Configuration

### Kafka (`config/kafka_config.py`)
- Bootstrap servers
- Topics configuration
- Producer/Consumer settings

### Settings généraux (`config/settings.yaml`)
- Paramètres d'API
- Seuils de sentiment
- Intervalles de collecte
