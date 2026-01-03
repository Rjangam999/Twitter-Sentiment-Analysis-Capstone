import sys 
import os 
sys.path.append(os.path.abspath("."))

import streamlit as st 
import joblib 

from src.data.preprocess import clean_text , adv_process
from src.utils.logger import get_logger 
from src.utils.config_loader import load_config 

# ---- Initializing logger & config
logger = get_logger('streamlit')
config = load_config()

# Load artifacts 
model = joblib.load(config['model']['model_path'])
vectorizer = joblib.load(config['model']['vectorizer_path'])

# -- Streamlit UI
st.set_page_config(page_title="Twitter Sentiment Analysis")

st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet to predict sentiment")

tweet = st.text_area("Tweet text")

# - Prediction
if st.button("Predict"):
    
    # -- check empty tweets.
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
        st.stop()

    # -- input length protection
    if len(tweet) > config['inference']['max_input_length']:
        st.warning("Tweet too long. Please shorten it..")
        st.stop()

    # -- try inference with logging + (try error protection )
    try:
        cleaned = adv_process(clean_text(tweet))
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]

        if prediction == 'positive':
            st.success("😊 Positive sentiment")
        else:
            st.error("😞 Negative sentiment")

    except Exception as e:
        logger.error(f"Inference failed. {e}")
        st.error('Something went wrong. Please try again.')
        st.stop()
