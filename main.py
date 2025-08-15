# app.py
import streamlit as st
import pickle
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# Load model & vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Clean text function (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Streamlit UI
st.title("🛍 Amazon Review Sentiment Analysis")
st.write("Enter a product review below to see if it's Positive, Neutral, or Negative.")

user_input = st.text_area("Your review:")

if st.button("Predict"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.subheader(f"Sentiment: **{prediction.capitalize()}**")
    else:
        st.warning("Please enter some text first.")
