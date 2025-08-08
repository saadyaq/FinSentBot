# 🧠 FinSentBot

**FinSentBot** est un projet de trading automatique basé sur l'analyse de sentiment en temps réel des actualités financières et des cours boursiers.  
Il utilise **Apache Kafka** pour gérer les flux de données, **Transformers NLP** pour analyser les sentiments des articles de presse, et une logique décisionnelle pour générer des signaux d'achat ou de vente.

---

## 📊 Objectif

Permettre à un modèle de :
- Lire des **prix d'actions** en temps réel (Yahoo Finance, etc.)
- Scraper des **articles financiers** (CNBC, Reuters…)
- Générer des **scores de sentiment**
- Produire un **signal Buy/Sell** basé sur ces informations
- Afficher le tout via un **dashboard interactif** (Streamlit)

---

## 🏗️ Architecture du projet

```bash
FinSentBot/
├── kafka/
│   ├── producers/
│   │   ├── stock_price_producer.py
│   │   └── news_scraper_producer.py
│   └── consumers/
│       ├── sentiment_analysis_consumer.py
│       └── trading_signal_consumer.py
│
├── nlp/
│   ├── sentiment_model.py
│   └── preprocessing.py
│
├── data/
│   ├── raw/
│   └── logs/
│
├── dashboard/
│   └── app.py
│
├── config/
│   ├── kafka_config.py
│   └── settings.yaml
│
├── utils/
│   └── helpers.py
│
├── requirements.txt
├── docker-compose.yml
├── .env
└── README.md

---

## 🚀 Démarrage rapide

1. Installez les dépendances Python :

```bash
pip install -r requirements.txt
```

2. Lancez l'infrastructure Kafka :

```bash
docker-compose up -d
```

3. Vérifiez la connexion en exécutant le script de test :

```bash
python test_send.py
```

Ce script envoie un message factice sur le topic `raw_news` afin de vérifier que
l'installation fonctionne correctement.

4. Générer un dataset d'entraînement à partir des données brutes :

```bash
python TradingLogic/prepare_dataset.py
```

Le fichier `data/training_datasets/train.csv` sera créé avec les échantillons alignés par symbole.
