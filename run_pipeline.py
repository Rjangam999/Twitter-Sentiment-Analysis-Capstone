# --- Import necessary libraries
import os 
import json 
import joblib 
import logging
import numpy as np 
import pandas as pd 

from src.data.preprocess import adv_process
from src.features.text_vectorizer import build_tfidf_vectorizer
from src.models.baseline_ml import train_logistic_regression
from src.evaluation.metrics import evaluate_classification 

# --- for model explainability 
import shap 
from src.explainability.shap_explainer import explain_logistic_regression

# ---- LSTM imports
from src.models.lstm_model import build_LSTM
from src.features.tokenizer import build_and_train_tokenizer, tokenize_and_pad

# --- centralized logging 
from src.utils.config_loader import load_config 
from src.utils.logger import get_logger 

logger = get_logger("pipelines")
config = load_config()

# ------- Baseline ML ( Logistic Regression ) model ---------------------------
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

# ----------- DL (LSTM) MODEL -------------------------------------
def train_lstm():
    from keras.callbacks import EarlyStopping

    logger.info("Starting LSTM training pipeline..")

    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')

    # ---- Preprocessing text for LSTM
    logger.info("Preprocessing text for LSTM")
        #  read_csv loads empty string as NaN 
    train_df['text'] = train_df['text'].fillna('')
    val_df['text'] = val_df['text'].fillna("")

    # ---- Labels
    y_train = (train_df['target'] == 'positive').astype(int)
    y_val = (val_df['target'] == 'positive').astype(int)

    # ---- Tokenization
    max_len= 50
    vocab_size = 20000

    logger.info("Building and Training Tokenizer...")
    tokenizer = build_and_train_tokenizer(
        train_df['text'],
        vocab_size=vocab_size,
        max_len=max_len)
    
    X_train = tokenize_and_pad( tokenizer, train_df['text'])
    X_val = tokenize_and_pad(tokenizer, val_df['text'])

    # ------ Model
    logger.info("Building LSTM...")
    model = build_LSTM(vocab_size, max_len)

    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    logger.info("Training LSTM model...")

    model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=4, 
        batch_size=64, 
        callbacks=[early_stop],
    )

    # ---- Save model (artifacts )
    logger.info("Saving Model...")
    model_dir = config['artifacts']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    
    model.save(f"{model_dir}/lstm_model.keras")
    joblib.dump(tokenizer, f"{model_dir}/lstm_tokenizer.pkl")

    logger.info("LSTM model and tokenizer saved successfully..")


if __name__ == "__main__":
    train_lstm()