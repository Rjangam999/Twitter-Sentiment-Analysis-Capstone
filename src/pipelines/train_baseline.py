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

# --- centralized logging 
from src.utils.config_loader import load_config 
from src.utils.logger import get_logger 

logger = get_logger("pipelines")
config = load_config()

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
    logger.info("Loading training and validation data")
    train_df = pd.read_csv(config['data']['train_path'], encoding='utf-8')
    val_df = pd.read_csv(config['data']['val_path'], encoding='utf-8')

    # --- Text preprocessing :
    logger.info("Applying lemmatization and stopwords removal with preprocessing...")
    train_df['text'] = train_df['text'].apply(adv_process)
    val_df['text'] = val_df['text'].apply(adv_process)

    # --- Vectorization 
    logger.info("Building TF-IDF vectorizer..")
    vectorizer = build_tfidf_vectorizer()
    x_train = vectorizer.fit_transform(train_df['text'])
    x_val = vectorizer.transform(val_df['text'])

    y_train = train_df['target']
    y_val = val_df['target']

    # ---- Model Training 
    logger.info("Training Logistic Regression Model")
    model = train_logistic_regression(x_train,y_train)

    # ---- Evaluation
    logger.info("Evaluating model on validation set..")
    preds = model.predict(x_val)
    report , matrix = evaluate_classification(y_val, preds)
    # print('Baseline validation metric Report :\n',report)
    # print('Confusion Matrix :\n', matrix)

    # ---- Save models (artifacts )
    model_dir = config["artifacts"]['model_dir']
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, f"{model_dir}/logistic_regression.pkl")
    joblib.dump(vectorizer, f"{model_dir}/tfidf_vectorizer.pkl")
    
    logger.info("Model and vectorizer saved succesfully..")

    # ----- explain model 
    logger.info("Generating SHAP explainability artifacts..")
    feature_names = vectorizer.get_feature_names_out()

    sample_idx = np.random.choice(x_val.shape[0], size=50, replace=False)
    x_sample = x_val[sample_idx]

    explain_logistic_regression(
        model, 
        x_sample, 
        feature_names 
    )

    # shap.summary_plot(shap_values, x_sample, feature_names=feature_names)
    logger.info("Training pipeline completed successfully....")



if __name__ == "__main__":
    baseline_ml()