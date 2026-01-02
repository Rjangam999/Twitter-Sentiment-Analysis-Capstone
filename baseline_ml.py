import pandas as pd 
import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ensures the NLTK resources downloads
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize resources 
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

from src.features.text_vectorizer import build_tfidf_vectorizer
from src.models.baseline_ml import train_logistic_regression
from src.evaluation.metrics import evaluate_classification 




def adv_process(text):

    # sometime csv loading convert empty strings to NaN(float), or None: 
    if not isinstance(text, str): 
        return ""
    
    # process text for baseline ML modeling
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words).strip()


def main():
    try:
        # ---- load data ----
        print('Loading Data...')
        train_df = pd.read_csv('data/processed/train.csv', encoding='utf-8')
        val_df = pd.read_csv('data/processed/val.csv', encoding='utf-8')

        # -- cleaning (adv_processing):
        print('Processing text (Lemmatization).....')

        train_df['text'] = train_df['text'].apply(adv_process)
        val_df['text'] = val_df['text'].apply(adv_process)

        # --- ML pipeline ---
        print('Building Vectorizer.....')

        vectorizer = build_tfidf_vectorizer()
        x_train = vectorizer.fit_transform(train_df['text'])
        x_val = vectorizer.transform(val_df['text'])

        y_train = train_df['target']
        y_val = val_df['target']

        print("Training Model...")
        model = train_logistic_regression(x_train,y_train)

        print('Evaluating.....')
        preds = model.predict(x_val)
        report , matrix = evaluate_classification(y_val, preds)

        print('Baseline validation metric Report :')
        print(report)
        print('Confusion Matrix :')
        print(matrix)

    except Exception as e:
        print(f"Error occured : {e}")



if __name__ == "__main__":
    main()
