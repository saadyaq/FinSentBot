from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentModel:

    def __init__(self):

        """Intialise le tokenizer et le modèle"""

        model_name="ProsusAI/finbert"
        print(f"Loading FinBert Model : {model_name}")
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.id2label = self.model.config.id2label
    
    def predict_sentiment(self,text):
        """ Prévoit le sentiment d'un texte nettoyé"""

        if not text:
            return 0.0
        
        inputs=self.tokenizer(
            text,
            return_tensors="pt",
            truncate=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():
            outputs=self.model(**inputs)
            logits=outputs.logits
            probas=torch.softmax(logits,dim=1).numpy()[0]
            
        label_probs = {
            self.id2label[str(i)]: prob
            for i, prob in enumerate(probas)
        }


        positive_score = label_probs.get("positive", 0)
        negative_score = label_probs.get("negative", 0)
        score = round(positive_score - negative_score, 4)
        return score
        