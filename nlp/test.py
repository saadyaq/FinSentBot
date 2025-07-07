from sentiment_model import SentimentModel
from preprocessing import clean_text

# ✅ Créer une instance du modèle
model = SentimentModel()

# ✅ Exemple de texte brut
raw_text = """
Market crashes after poor earnings reports!!
"""

# ✅ Nettoyer le texte
cleaned = clean_text(raw_text)
print(f"✅ Cleaned text: {cleaned}")

# ✅ Prédire le score
score,label_probs = model.predict_sentiment(cleaned)
print(f"✅ Label probabilities: {label_probs}")
print(f"✅ Predicted sentiment score: {score}")

