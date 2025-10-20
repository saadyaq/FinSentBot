#!/usr/bin/env python3
"""
Script de configuration pour l'entraînement du modèle de trading
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Exécuter une commande avec gestion d'erreur"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Terminé")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Erreur: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def check_requirements():
    """Vérifier que les dépendances sont installées"""
    print("📋 Vérification des dépendances...")

    required_packages = [
        'torch', 'transformers', 'sklearn',
        'pandas', 'numpy', 'matplotlib', 'seaborn'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - manquant")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️ Packages manquants: {', '.join(missing_packages)}")
        print("Installez-les avec: pip install -r requirements.txt")
        return False

    print("✅ Toutes les dépendances sont présentes")
    return True

def check_dataset():
    """Vérifier que le dataset existe"""
    dataset_path = Path("data/training_datasets/train_enhanced_with_historical.csv")

    if dataset_path.exists():
        print(f"✅ Dataset trouvé: {dataset_path}")

        # Informations sur le dataset
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            print(f"  📊 {len(df)} échantillons")
            print(f"  📈 Classes: {df['action'].value_counts().to_dict()}")
        except Exception as e:
            print(f"  ⚠️ Erreur lors de la lecture: {e}")

        return True
    else:
        print(f"❌ Dataset non trouvé: {dataset_path}")
        print("Générez le dataset avec: python TradingLogic/prepare_dataset.py")
        return False

def check_directories():
    """Créer les répertoires nécessaires"""
    print("📁 Vérification des répertoires...")

    directories = [
        "models",
        "data/training_datasets",
        "logs"
    ]

    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  📁 Créé: {directory}")
        else:
            print(f"  ✅ Existe: {directory}")

    return True

def test_model_import():
    """Tester l'import du modèle"""
    print("🧪 Test d'import du modèle...")

    try:
        sys.path.insert(0, str(Path.cwd()))
        from TradingLogic.trading_model import TradingSignalModel
        print("✅ Import du modèle réussi")
        return True
    except Exception as e:
        print(f"❌ Erreur d'import: {e}")
        return False

def quick_training_test():
    """Test rapide d'entraînement avec peu de données"""
    print("🚀 Test rapide d'entraînement...")

    # Commande de test avec échantillon réduit
    command = """
    cd TradingLogic && python run_training.py \
        --sample-size 100 \
        --epochs 1 \
        --batch-size 4 \
        --max-length 128
    """

    return run_command(command, "Test d'entraînement rapide")

def main():
    """Fonction principale de configuration"""
    print("🛠️ Configuration de l'environnement de trading")
    print("=" * 50)

    # Vérifications de base
    checks = [
        ("Dépendances", check_requirements),
        ("Répertoires", check_directories),
        ("Dataset", check_dataset),
        ("Import modèle", test_model_import)
    ]

    all_passed = True
    for name, check_func in checks:
        print(f"\n🔍 {name}:")
        if callable(check_func):
            result = check_func()
        else:
            result = check_func

        if not result:
            all_passed = False

    print("\n" + "=" * 50)

    if all_passed:
        print("🎉 Configuration terminée avec succès!")
        print("\nVous pouvez maintenant:")
        print("1. Entraîner le modèle: cd TradingLogic && python run_training.py")
        print("2. Test rapide: cd TradingLogic && python run_training.py --sample-size 1000 --epochs 3")
        print("3. Tester le modèle: cd TradingLogic && python test_model.py")

        # Proposer un test rapide
        response = input("\n🤔 Voulez-vous lancer un test d'entraînement rapide? (y/n): ")
        if response.lower() in ['y', 'yes', 'oui']:
            quick_training_test()
    else:
        print("⚠️ Configuration incomplète. Veuillez corriger les erreurs ci-dessus.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())