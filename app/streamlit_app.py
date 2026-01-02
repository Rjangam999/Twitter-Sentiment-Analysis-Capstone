import sys 
import os 
sys.path.append(os.path.abspath("."))

import streamlit as st 
import joblib 

from src.data.preprocess import clean_text , adv_process

# Load artifacts 
model = joblib.load("models/logistic_regression.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.set_page_config(page_title="Twitter Sentiment Analysis")

st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet to predict sentiment")

tweet = st.text_area("Tweet text")

if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        cleaned = adv_process(clean_text(tweet))
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]

        if prediction == 'positive':
            st.success("😊 Positive sentiment")
        else:
            st.error("😞 Negative sentiment")
