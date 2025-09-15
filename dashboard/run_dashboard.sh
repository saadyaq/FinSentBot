#!/bin/bash

# Script de lancement du dashboard FinSentBot

echo "🚀 Lancement du dashboard FinSentBot..."

# Vérifier que nous sommes dans le bon répertoire
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 Répertoire du projet: $PROJECT_ROOT"

# Activer l'environnement virtuel si il existe
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "🔧 Activation de l'environnement virtuel..."
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Installer les dépendances si nécessaire
echo "📦 Vérification des dépendances..."
pip install -q -r "$PROJECT_ROOT/requirements.txt"

# Lancer le dashboard
echo "🎯 Lancement du dashboard sur http://localhost:8501"
cd "$SCRIPT_DIR"
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

echo "✅ Dashboard fermé"