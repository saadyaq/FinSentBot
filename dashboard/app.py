from pathlib import Path
import sys

BASE_PATH = Path(__file__).resolve().parents[1]
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import os
from typing import Dict, Optional

import yfinance as yf
import requests

from TradingLogic.SignalGenerator.signal_generator import SignalGenerator, TradingSignal

# Configuration de la page
st.set_page_config(
    page_title="FinSentBot Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = BASE_PATH / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
TRAINING_DATA_PATH = DATA_PATH / "training_datasets"
INFERENCE_ENDPOINT = os.getenv("PREDICTION_API_URL")
HEALTH_ENDPOINT = os.getenv("PREDICTION_HEALTH_URL")

if INFERENCE_ENDPOINT:
    INFERENCE_ENDPOINT = INFERENCE_ENDPOINT.rstrip("/")
    if not HEALTH_ENDPOINT:
        if INFERENCE_ENDPOINT.endswith("/predict"):
            HEALTH_ENDPOINT = INFERENCE_ENDPOINT.rsplit("/", 1)[0] + "/health"
        else:
            HEALTH_ENDPOINT = f"{INFERENCE_ENDPOINT}/health"

@st.cache_data
def load_training_data():
    """Charger le dataset d'entraînement"""
    try:
        df = pd.read_csv(TRAINING_DATA_PATH / "train.csv")
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données d'entraînement: {e}")
        return pd.DataFrame()

@st.cache_data
def load_news_sentiment():
    """Charger les données de sentiment des news"""
    try:
        news_data = []
        with open(RAW_DATA_PATH / "news_sentiment.jsonl", 'r') as f:
            for line in f:
                news_data.append(json.loads(line))
        df = pd.DataFrame(news_data)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], errors="coerce", utc=True
            ).dt.tz_convert(None)
            df = df.dropna(subset=["timestamp"])
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données de sentiment: {e}")
        return pd.DataFrame()

@st.cache_data
def load_stock_prices():
    """Charger les prix des actions"""
    try:
        price_data = []
        with open(RAW_DATA_PATH / "stock_prices.jsonl", 'r') as f:
            for line in f:
                price_data.append(json.loads(line))
        df = pd.DataFrame(price_data)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], errors="coerce", utc=True
            ).dt.tz_convert(None)
            df = df.dropna(subset=["timestamp"])
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des prix: {e}")
        return pd.DataFrame()

@st.cache_data
def discover_model_checkpoints(base_dir: Path | str = BASE_PATH / "models" / "signal_generator") -> list[str]:
    """Retourne la liste des checkpoints signal_generator disponibles."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    checkpoints = sorted(base_path.glob("**/signal_generator.pth"))
    return [str(path.resolve()) for path in checkpoints]

@st.cache_resource
def load_signal_generator_cached(model_path: str, confidence_threshold: float) -> SignalGenerator:
    """Charge et met en cache le générateur de signaux."""
    return SignalGenerator(model_path, confidence_threshold=confidence_threshold)


@st.cache_data(show_spinner=False)
def fetch_api_health(url: str) -> Optional[dict]:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def call_prediction_api(url: str, payload: dict) -> dict:
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()

def fetch_market_snapshot(symbol: str) -> Dict[str, float]:
    """Récupère un instantané de marché basique via yfinance."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="5d", interval="1d")
    if hist.empty:
        raise ValueError(f"Aucune donnée trouvée pour {symbol}")
    price_now = float(hist["Close"].iloc[-1])
    previous_price = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price_now
    variation = (
        (price_now - previous_price) / previous_price if previous_price else 0.0
    )
    market_timestamp = hist.index[-1].to_pydatetime()
    return {
        "price": price_now,
        "previous_price": previous_price,
        "variation": variation,
        "price_future": price_now * (1 + variation),
        "timestamp": market_timestamp.isoformat(),
    }

def main():
    st.title("FinSentBot Dashboard")
    st.markdown("Dashboard de visualisation pour l'analyse de sentiment financier et les signaux de trading")

    # Sidebar pour navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choisir une page:",
        [
            "Vue d'ensemble",
            "Signaux de Trading",
            "Analyse de Sentiment",
            "Performance",
            "Données Temps Réel",
            "Générateur IA",
        ],
    )

    # Charger les données
    training_data = load_training_data()
    news_data = load_news_sentiment()
    price_data = load_stock_prices()

    if page == "Vue d'ensemble":
        overview_page(training_data, news_data, price_data)
    elif page == "Signaux de Trading":
        trading_signals_page(training_data)
    elif page == "Analyse de Sentiment":
        sentiment_analysis_page(news_data, training_data)
    elif page == "Performance":
        performance_page(training_data)
    elif page == "Données Temps Réel":
        realtime_data_page(price_data, news_data)
    elif page == "Générateur IA":
        ai_signal_generator_page()

def overview_page(training_data, news_data, price_data):
    """Page vue d'ensemble"""
    st.header("Vue d'ensemble")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Articles analysés", len(news_data) if not news_data.empty else 0)
    
    with col2:
        st.metric("Actions surveillées", len(training_data['symbol'].unique()) if not training_data.empty else 0)
    
    with col3:
        st.metric("Signaux générés", len(training_data) if not training_data.empty else 0)
    
    with col4:
        avg_sentiment = training_data['sentiment_score'].mean() if not training_data.empty else 0
        st.metric("Sentiment moyen", f"{avg_sentiment:.3f}")

    if not training_data.empty:
        # Distribution des actions
        st.subheader("Distribution des signaux de trading")
        action_counts = training_data['action'].value_counts()
        fig_actions = px.pie(
            values=action_counts.values, 
            names=action_counts.index,
            title="Répartition des signaux BUY/SELL/HOLD",
            color_discrete_map={'BUY': '#00CC96', 'SELL': '#EF553B', 'HOLD': '#FFA15A'}
        )
        st.plotly_chart(fig_actions, use_container_width=True)

        # Top symbols
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Actions les plus analysées")
            top_symbols = training_data['symbol'].value_counts().head(10)
            fig_symbols = px.bar(
                x=top_symbols.values, 
                y=top_symbols.index,
                orientation='h',
                title="Top 10 des actions"
            )
            st.plotly_chart(fig_symbols, use_container_width=True)
        
        with col2:
            st.subheader("Distribution du sentiment")
            fig_sentiment = px.histogram(
                training_data, 
                x='sentiment_score', 
                nbins=50,
                title="Distribution des scores de sentiment"
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

def trading_signals_page(training_data):
    """Page signaux de trading"""
    st.header("Signaux de Trading")
    
    if training_data.empty:
        st.warning("Aucune donnée de trading disponible")
        return

    # Filtres
    col1, col2 = st.columns(2)
    with col1:
        selected_symbol = st.selectbox("Filtrer par symbole:", ['Tous'] + list(training_data['symbol'].unique()))
    with col2:
        selected_action = st.selectbox("Filtrer par action:", ['Tous', 'BUY', 'SELL', 'HOLD'])

    # Application des filtres
    filtered_data = training_data.copy()
    if selected_symbol != 'Tous':
        filtered_data = filtered_data[filtered_data['symbol'] == selected_symbol]
    if selected_action != 'Tous':
        filtered_data = filtered_data[filtered_data['action'] == selected_action]

    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Relation Sentiment vs Variation de Prix")
        fig_scatter = px.scatter(
            filtered_data,
            x='sentiment_score',
            y='variation',
            color='action',
            hover_data=['symbol', 'price_now'],
            title="Sentiment vs Variation de Prix",
            color_discrete_map={'BUY': '#00CC96', 'SELL': '#EF553B', 'HOLD': '#FFA15A'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.subheader("Distribution des variations par action")
        fig_box = px.box(
            filtered_data,
            x='action',
            y='variation',
            title="Distribution des variations",
            color='action',
            color_discrete_map={'BUY': '#00CC96', 'SELL': '#EF553B', 'HOLD': '#FFA15A'}
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Tableau des signaux récents
    st.subheader("Signaux récents")
    if not filtered_data.empty:
        display_cols = ['symbol', 'action', 'sentiment_score', 'price_now', 'variation']
        st.dataframe(filtered_data[display_cols].head(20))

def sentiment_analysis_page(news_data, training_data):
    """Page analyse de sentiment"""
    st.header("Analyse de Sentiment")
    
    if news_data.empty and training_data.empty:
        st.warning("Aucune donnée de sentiment disponible")
        return

    # Utiliser les données de news si disponibles, sinon les données d'entraînement
    data_to_use = news_data if not news_data.empty else training_data
    
    # Évolution du sentiment dans le temps
    if 'timestamp' in data_to_use.columns:
        st.subheader("Évolution du sentiment dans le temps")
        
        # Grouper par jour
        data_to_use["date"] = pd.to_datetime(
            data_to_use["timestamp"], errors="coerce"
        ).dt.date
        data_to_use = data_to_use.dropna(subset=["date"])
        daily_sentiment = (
            data_to_use.groupby("date")["sentiment_score"]
            .agg(["mean", "count"])
            .reset_index()
        )
        
        fig_timeline = px.line(
            daily_sentiment,
            x='date',
            y='mean',
            title="Évolution du sentiment moyen par jour"
        )
        fig_timeline.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_timeline, use_container_width=True)

    # Analyse par source (si disponible)
    if 'source' in data_to_use.columns:
        st.subheader("Sentiment par source")
        source_sentiment = data_to_use.groupby('source')['sentiment_score'].agg(['mean', 'count']).reset_index()
        source_sentiment = source_sentiment.sort_values('mean', ascending=True)
        
        fig_source = px.bar(
            source_sentiment,
            x='mean',
            y='source',
            orientation='h',
            title="Sentiment moyen par source d'information"
        )
        st.plotly_chart(fig_source, use_container_width=True)

    # Mots-clés les plus fréquents (analyse simplifiée)
    if 'content' in data_to_use.columns or 'text' in data_to_use.columns:
        st.subheader("Analyse textuelle")
        text_column = 'content' if 'content' in data_to_use.columns else 'text'
        
        # Sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                data_to_use,
                x='sentiment_score',
                nbins=30,
                title="Distribution des scores de sentiment"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Catégoriser les sentiments
            data_to_use['sentiment_category'] = pd.cut(
                data_to_use['sentiment_score'],
                bins=[-1, -0.1, 0.1, 1],
                labels=['Négatif', 'Neutre', 'Positif']
            )
            sentiment_counts = data_to_use['sentiment_category'].value_counts()
            
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Répartition des catégories de sentiment"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

def performance_page(training_data):
    """Page performance"""
    st.header("Performance du Modèle")
    
    if training_data.empty:
        st.warning("Aucune donnée de performance disponible")
        return

    # Métriques de performance
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Précision par action
        action_counts = training_data['action'].value_counts()
        st.metric("Signal le plus fréquent", action_counts.index[0])
    
    with col2:
        # Correlation sentiment-variation
        if 'sentiment_score' in training_data.columns and 'variation' in training_data.columns:
            correlation = training_data['sentiment_score'].corr(training_data['variation'])
            st.metric("Corrélation Sentiment-Variation", f"{correlation:.3f}")
    
    with col3:
        # Variation moyenne positive vs négative
        positive_sentiment = training_data[training_data['sentiment_score'] > 0]['variation'].mean()
        st.metric("Variation moy. (sentiment +)", f"{positive_sentiment:.3f}%")

    # Matrice de confusion simulée
    st.subheader("Analyse des signaux par sentiment")
    
    # Créer des catégories de sentiment
    training_data['sentiment_category'] = pd.cut(
        training_data['sentiment_score'],
        bins=[-1, -0.1, 0.1, 1],
        labels=['Négatif', 'Neutre', 'Positif']
    )
    
    # Table croisée
    crosstab = pd.crosstab(training_data['sentiment_category'], training_data['action'])
    
    # Heatmap
    fig_heatmap = px.imshow(
        crosstab.values,
        x=crosstab.columns,
        y=crosstab.index,
        aspect="auto",
        title="Matrice Sentiment vs Action",
        color_continuous_scale="RdYlGn"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Performance par symbole
    st.subheader("Performance par action")
    symbol_performance = training_data.groupby('symbol').agg({
        'sentiment_score': 'mean',
        'variation': 'mean',
        'action': 'count'
    }).reset_index()
    symbol_performance.columns = ['Symbol', 'Sentiment Moyen', 'Variation Moyenne', 'Nombre Signaux']
    symbol_performance = symbol_performance.sort_values('Nombre Signaux', ascending=False).head(20)
    
    st.dataframe(symbol_performance)

def realtime_data_page(price_data, news_data):
    """Page données temps réel"""
    st.header("Données Temps Réel")
    
    # Auto-refresh
    if st.checkbox("Auto-refresh (30s)"):
        st.rerun()

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prix des Actions")
        if not price_data.empty:
            # Derniers prix
            latest_prices = price_data.sort_values('timestamp').groupby('symbol').tail(1)
            st.dataframe(latest_prices[['symbol', 'price', 'timestamp']])
            
            # Graphique des prix récents pour un symbole sélectionné
            if not latest_prices.empty:
                selected_stock = st.selectbox("Sélectionner une action:", latest_prices['symbol'].unique())
                stock_data = price_data[price_data['symbol'] == selected_stock].sort_values('timestamp')
                
                if len(stock_data) > 1:
                    fig_price = px.line(
                        stock_data,
                        x='timestamp',
                        y='price',
                        title=f"Évolution du prix - {selected_stock}"
                    )
                    st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info("Aucune donnée de prix en temps réel disponible")
    
    with col2:
        st.subheader("News Récentes")
        if not news_data.empty:
            # Dernières news
            recent_news = news_data.sort_values('timestamp', ascending=False).head(10)
            for _, news in recent_news.iterrows():
                with st.expander(f"{news.get('source', 'Unknown')} - {news.get('title', 'No title')[:50]}..."):
                    st.write(f"**Sentiment:** {news.get('sentiment_score', 'N/A'):.3f}")
                    st.write(f"**Symbole:** {news.get('symbol', 'N/A')}")
                    st.write(f"**Date:** {news.get('timestamp', 'N/A')}")
                    if 'content' in news and news['content']:
                        st.write(news['content'][:200] + "...")
        else:
            st.info("Aucune news récente disponible")

    # Alertes et notifications
    st.subheader("Alertes")
    if not price_data.empty:
        # Exemple d'alertes basées sur les données
        st.info("Système d'alertes à implémenter")
    else:
        st.warning("Connectez les données en temps réel pour activer les alertes")

def ai_signal_generator_page():
    """Interface de génération de signaux en direct."""
    st.header("Générateur de Signaux IA")

    use_remote = bool(INFERENCE_ENDPOINT)
    generator: Optional[SignalGenerator] = None
    model_metadata: dict = {}

    if use_remote:
        st.info("Mode API externe activé (PREDICTION_API_URL).")
        if HEALTH_ENDPOINT:
            metadata = fetch_api_health(HEALTH_ENDPOINT)
            if metadata is None:
                st.error("Impossible de contacter l'API d'inférence (endpoint santé).")
                return
            model_metadata = metadata
        else:
            st.warning("Aucun endpoint santé spécifié ; les métadonnées du modèle ne seront pas affichées.")
    else:
        checkpoints = discover_model_checkpoints()
        if not checkpoints:
            st.warning("Aucun modèle signal_generator.pth trouvé dans models/signal_generator.")
            st.info("Lancez un entraînement ou configurez PREDICTION_API_URL pour utiliser un endpoint distant.")
            return

        checkpoint_labels = []
        for path_str in checkpoints:
            try:
                rel = Path(path_str).relative_to(BASE_PATH)
                label = str(rel)
            except ValueError:
                label = path_str
            checkpoint_labels.append(label)

        newest_index = max(
            range(len(checkpoints)), key=lambda idx: Path(checkpoints[idx]).stat().st_mtime
        )

        selected_label = st.selectbox(
            "Checkpoint du modèle",
            checkpoint_labels,
            index=newest_index,
        )
        selected_path = checkpoints[checkpoint_labels.index(selected_label)]

        confidence_threshold = st.slider(
            "Seuil de confiance minimum",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
        )

        try:
            generator = load_signal_generator_cached(selected_path, confidence_threshold)
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Erreur lors du chargement du modèle: {exc}")
            return

        if st.session_state.get("last_model_path") != selected_path:
            st.session_state["last_prediction"] = None
        st.session_state["last_model_path"] = selected_path

        model_metadata = {
            "model_type": generator.model_type.upper(),
            "feature_count": len(generator.feature_cols),
            "sequence_length": generator.sequence_length,
            "feature_cols": generator.feature_cols,
        }

    info_cols = st.columns(3)
    if "model_type" in model_metadata:
        info_cols[0].metric("Type de modèle", str(model_metadata["model_type"]))
    if "feature_count" in model_metadata:
        info_cols[1].metric("Nombre de features", model_metadata["feature_count"])
    if "sequence_length" in model_metadata:
        info_cols[2].metric("Longueur séquence", model_metadata["sequence_length"])

    if use_remote:
        if st.session_state.get("last_model_path") != "api":
            st.session_state["last_prediction"] = None
        st.session_state["last_model_path"] = "api"
    elif generator:
        with st.expander("Détails des features attendus"):
            st.write(", ".join(generator.feature_cols))
    elif use_remote and model_metadata.get("feature_cols"):
        with st.expander("Détails des features (API)"):
            st.write(", ".join(model_metadata["feature_cols"]))

    default_market = {
        "price": 0.0,
        "previous_price": 0.0,
        "variation": 0.0,
        "price_future": 0.0,
        "sentiment_score": 0.0,
        "timestamp": "",
    }
    if "market_data" not in st.session_state:
        st.session_state["market_data"] = default_market.copy()
    if "market_symbol" not in st.session_state:
        st.session_state["market_symbol"] = "AAPL"

    symbol = st.text_input(
        "Symbole boursier (ex: AAPL)",
        value=st.session_state["market_symbol"],
        max_chars=10,
    ).upper().strip()
    st.session_state["market_symbol"] = symbol if symbol else st.session_state["market_symbol"]

    fetch_col, sentiment_col = st.columns([1, 1])
    with fetch_col:
        if st.button("📈 Charger les données de marché (yfinance)"):
            try:
                snapshot = fetch_market_snapshot(symbol)
                st.session_state["market_data"].update(snapshot)
                st.success(
                    f"Données mises à jour pour {symbol} (timestamp {snapshot.get('timestamp')})."
                )
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Impossible de récupérer les données de marché: {exc}")

    with sentiment_col:
        sentiment_score = st.slider(
            "Score de sentiment",
            min_value=-1.0,
            max_value=1.0,
            value=float(st.session_state["market_data"].get("sentiment_score", 0.0)),
            step=0.01,
        )

    market_data = st.session_state["market_data"]

    price_col, prev_price_col, variation_col = st.columns(3)
    current_price = price_col.number_input(
        "Prix actuel ($)",
        value=float(market_data.get("price", 0.0)),
        step=0.1,
        format="%.2f",
    )
    previous_price = prev_price_col.number_input(
        "Prix précédent ($)",
        value=float(
            market_data.get("previous_price", current_price if current_price else 0.0)
        ),
        step=0.1,
        format="%.2f",
    )
    auto_variation = (
        (current_price - previous_price) / previous_price if previous_price else 0.0
    )
    variation = variation_col.number_input(
        "Variation (fraction)",
        value=float(market_data.get("variation", auto_variation)),
        step=0.0001,
        format="%.4f",
    )

    future_price = st.number_input(
        "Prix futur estimé ($)",
        value=float(market_data.get("price_future", current_price * (1 + variation))),
        step=0.1,
        format="%.2f",
    )

    market_data.update(
        {
            "price": current_price,
            "previous_price": previous_price,
            "variation": variation,
            "price_future": future_price,
            "sentiment_score": sentiment_score,
        }
    )

    st.caption(
        "La variation est exprimée en fraction (ex: 0.025 = +2.5%). "
        "Vous pouvez ajuster manuellement les valeurs pour simuler différents scénarios."
    )

    generate = st.button("🎯 Générer le signal IA")

    if generate:
        if not symbol:
            st.error("Veuillez renseigner un symbole valide.")
            return

        try:
            if use_remote:
                if not INFERENCE_ENDPOINT:
                    st.error("Endpoint API non configuré.")
                    return
                payload_market = {
                    "sentiment_score": sentiment_score,
                    "price": current_price,
                    "previous_price": previous_price if previous_price else None,
                    "variation": variation,
                    "price_future": future_price,
                    "extra": {},
                }
                known_keys = set(payload_market.keys()) | {"sentiment_score"}
                extras = {
                    key: value
                    for key, value in market_data.items()
                    if key not in known_keys
                }
                if extras:
                    payload_market["extra"] = extras
                api_payload = {
                    "symbol": symbol,
                    "market_data": payload_market,
                }
                response = call_prediction_api(INFERENCE_ENDPOINT, api_payload)
                signal = TradingSignal(**response)
            else:
                signal = generator.generate_signal(symbol, market_data)
            st.session_state["last_prediction"] = signal
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Erreur lors de la génération du signal: {exc}")
            return

    if st.session_state.get("last_prediction"):
        signal = st.session_state["last_prediction"]

        st.subheader("Résultat du modèle")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Action suggérée", signal.action)
        with metric_col2:
            st.metric("Confiance", f"{signal.confidence * 100:.1f}%")
        with metric_col3:
            st.metric("Prix utilisé", f"${signal.price:.2f}")

        st.markdown(f"**Raisonnement du modèle :** {signal.reasoning}")

        extra_cols = st.columns(3)
        with extra_cols[0]:
            if signal.stop_loss:
                st.metric("Stop Loss", f"${signal.stop_loss:.2f}")
        with extra_cols[1]:
            if signal.take_profit:
                st.metric("Take Profit", f"${signal.take_profit:.2f}")
        with extra_cols[2]:
            if signal.position_size:
                st.metric("Taille de position", f"{signal.position_size:.2f} unités")

        if generator.feature_cols:
            raw_features = {
                col: market_data.get(col, 0.0) for col in generator.feature_cols
            }
            scaled_values = generator.scaler.transform(
                np.array([list(raw_features.values())], dtype=np.float32)
            )[0]
            features_df = pd.DataFrame(
                {
                    "Feature": generator.feature_cols,
                    "Valeur": [raw_features[col] for col in generator.feature_cols],
                    "Normalisée": scaled_values,
                }
            )
            st.subheader("Features utilisés par le modèle")
            st.dataframe(features_df, use_container_width=True)

        st.caption(
            "Les niveaux de stop loss / take profit sont fournis par la logique du modèle SignalGenerator."
        )

        active_threshold = (
            model_metadata.get("confidence_threshold")
            if use_remote
            else confidence_threshold
        )
        if active_threshold is not None:
            st.success(
                f"Signal généré à {signal.timestamp}. Seuil utilisé : {float(active_threshold):.2f}"
            )
        else:
            st.success(f"Signal généré à {signal.timestamp}.")

if __name__ == "__main__":
    main()
