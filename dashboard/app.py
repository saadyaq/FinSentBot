import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import os

# Configuration de la page
st.set_page_config(
    page_title="FinSentBot Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration des chemins
BASE_PATH = "/home/saadyaq/SE/Python/finsentbot"
DATA_PATH = f"{BASE_PATH}/data"
RAW_DATA_PATH = f"{DATA_PATH}/raw"
TRAINING_DATA_PATH = f"{DATA_PATH}/training_datasets"

@st.cache_data
def load_training_data():
    """Charger le dataset d'entraînement"""
    try:
        df = pd.read_csv(f"{TRAINING_DATA_PATH}/train.csv")
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données d'entraînement: {e}")
        return pd.DataFrame()

@st.cache_data
def load_news_sentiment():
    """Charger les données de sentiment des news"""
    try:
        news_data = []
        with open(f"{RAW_DATA_PATH}/news_sentiment.jsonl", 'r') as f:
            for line in f:
                news_data.append(json.loads(line))
        df = pd.DataFrame(news_data)
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données de sentiment: {e}")
        return pd.DataFrame()

@st.cache_data
def load_stock_prices():
    """Charger les prix des actions"""
    try:
        price_data = []
        with open(f"{RAW_DATA_PATH}/stock_prices.jsonl", 'r') as f:
            for line in f:
                price_data.append(json.loads(line))
        df = pd.DataFrame(price_data)
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des prix: {e}")
        return pd.DataFrame()

def main():
    st.title("FinSentBot Dashboard")
    st.markdown("Dashboard de visualisation pour l'analyse de sentiment financier et les signaux de trading")

    # Sidebar pour navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choisir une page:",
        ["Vue d'ensemble", "Signaux de Trading", "Analyse de Sentiment", "Performance", "Données Temps Réel"]
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
        data_to_use['date'] = pd.to_datetime(data_to_use['timestamp']).dt.date
        daily_sentiment = data_to_use.groupby('date')['sentiment_score'].agg(['mean', 'count']).reset_index()
        
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

if __name__ == "__main__":
    main()