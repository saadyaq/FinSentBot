from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SentimentModel:

    def __init__(self):

        """Intialise le tokenizer et le modèle"""

        model_name="ProsusAI/finbert"
        try:
            logger.info(f"Loading FinBERT model: {model_name}")
            self.tokenizer=AutoTokenizer.from_pretrained(model_name)
            self.model=AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            self.id2label = self.model.config.id2label
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise
    
    def predict_sentiment(self,text):
        """ Prévoit le sentiment d'un texte nettoyé"""
        try:
            if not text:
                logger.warning("Empty text provided for sentiment analysis")
                return 0.0
            
            inputs=self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding='max_length',
                max_length=256
            )
        except Exception as e:
            logger.error(f"Failed to tokenize text for sentiment analysis: {e}")
            return 0.0

        with torch.no_grad():
            outputs=self.model(**inputs)
            logits=outputs.logits
            probas=torch.softmax(logits,dim=1).numpy()[0]
            
        label_probs = {
            self.id2label[i]: prob
            for i, prob in enumerate(probas)
        }


        positive_score = label_probs.get("positive", 0)
        negative_score = label_probs.get("negative", 0)
        score = round(positive_score - negative_score, 4)
        return score
        