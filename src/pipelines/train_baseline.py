# --- Import necessary libraries

import os 
import json 
import joblib 
import logging
import numpy as np 
import pandas as pd 

from src.features.text_vectorizer import build_tfidf_vectorizer
from src.models.baseline_ml import train_logistic_regression
from src.evaluation.metrics import evaluate_classification 

# --- for model explainability 
import shap 
from src.explainability.shap_explainer import explain_logistic_regression

import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ----- ensures the NLTK resources downloads
# nltk.download('wordnet', quiet=True)
# nltk.download('stopwords', quiet=True)

# Initialize resources 
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def adv_process(text):

    # sometime csv loading convert empty strings to NaN(float), or None: 
    if not isinstance(text, str): 
        return ""
    
    # process text for baseline ML modeling
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words).strip()


def baseline_ml():

    # ---- load data ----
    print('Loading Data...')
    train_df = pd.read_csv('data/processed/train.csv', encoding='utf-8')
    val_df = pd.read_csv('data/processed/val.csv', encoding='utf-8')

    # -- cleaning (adv_processing):
    print('Processing text (Lemmatization).....')
    train_df['text'] = train_df['text'].apply(adv_process)
    val_df['text'] = val_df['text'].apply(adv_process)

    # --- Vectorization 
    print('Building Vectorizer.....')
    vectorizer = build_tfidf_vectorizer()
    x_train = vectorizer.fit_transform(train_df['text'])
    x_val = vectorizer.transform(val_df['text'])

    y_train = train_df['target']
    y_val = val_df['target']

    # ---- Model Training 
    print("Training Model...")
    model = train_logistic_regression(x_train,y_train)

    # ---- Evaluation
    print('Evaluating.....')
    preds = model.predict(x_val)
    report , matrix = evaluate_classification(y_val, preds)

    print('Baseline validation metric Report :\n',report)
    print('Confusion Matrix :\n', matrix)

    # ---- Save models (artifacts )
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/logistic_regression.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print('Model and Vectorizer Saved successfully..')

    # ----- explain model 
    print('Logistic Regression Explainability plots....')
    feature_names = vectorizer.get_feature_names_out()
    sample_idx = np.random.choice(x_val.shape[0], size=50, replace=False)
    x_sample = x_val[sample_idx]

    shap_values = explain_logistic_regression(
        model, 
        x_sample, 
        feature_names 
    )

    # shap.summary_plot(shap_values, x_sample, feature_names=feature_names)


if __name__ == "__main__":
    baseline_ml()