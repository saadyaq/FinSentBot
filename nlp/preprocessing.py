import re 


#Cleaning the text from miscelasneous

def clean_text(text:str)->str:

    """
        Nettoyer les texts avant d'alimenter le modèle NLP.
    """

    if not text :

        return ""
    #Mettre le texte en miniscule 

    text=text.lower()

    #Enlever les balises HTML
    text=re.sub(r"<.!*?>"," ",text) 

    #Enlever les urls
    text=re.sub(r"http\S+"," ",text)

    #Enlever les caractères alphanumériques
    text=re.sub(r"[^a-zA-Z0-9.,;!?()'\"]"," ",text)

    #Enlever les caractères spéciaux s'il dépasse une occurence cote à cote
    text= re.sub(r"([!?.,])\1+", r"\1", text)

    #Enlever les espaces multiples
    text=re.sub(r"\s+"," ",text).strip()

    return text
