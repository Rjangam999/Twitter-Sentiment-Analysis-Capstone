import re 
import string 

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