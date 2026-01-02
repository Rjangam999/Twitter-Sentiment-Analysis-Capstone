import re 
import string 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str: 
    # lower case 
    text = text.lower()
    # URLs removal 
    text = re.sub(r'http\S+', '', text)
    # @mentions removal 
    text = re.sub('@\w+','',text)
    # hastag removal 
    text = re.sub(r'#\w+', '', text)    
    # remove panctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove digits 
    text = re.sub(r'\d+', '' , text)
    # remove white spaces 
    text = text.strip()

    return text 

def adv_process(text):

    # sometime csv loading convert empty strings to NaN(float), or None: 
    if not isinstance(text, str): 
        return ""
    
    # process text for baseline ML modeling
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words).strip()