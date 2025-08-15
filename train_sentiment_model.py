# train_sentiment_model.py
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download NLTK stopwords
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# 1. Load Dataset
df = pd.read_csv("C:\\Users\\kunal\\PycharmProjects\\customer_review_analysis\\data\\Reviews.csv")
  # Download from Kaggle first
df = df[["Score", "Text"]].dropna()

# 2. Map Scores to Sentiment
def score_to_sentiment(score):
    if score <= 2:
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"

df["Sentiment"] = df["Score"].apply(score_to_sentiment)

# 3. Clean Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

df["CleanText"] = df["Text"].apply(clean_text)

# 4. Features & Labels
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["CleanText"])
y = df["Sentiment"]

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 8. Save Model & Vectorizer
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved!")
