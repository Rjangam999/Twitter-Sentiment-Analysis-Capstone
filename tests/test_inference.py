import joblib 
from src.data.preprocess import clean_text , adv_process

def test_model_inference_runs():
    model = joblib.load("models/logistic_regression.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    text = "I love this product."
    cleaned = adv_process(clean_text(text))
    features = vectorizer.transform([cleaned])

    prediction = model.predict(features)

    assert prediction[0] in ['positive', 'negative']