import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

st.title("🎬 Movie Review Sentiment Analysis")

# ---------- CACHE DATASET ----------
@st.cache_data
def load_data():
    data = pd.read_csv("IMDB Dataset.csv")
    data['review'] = data['review'].str.lower()
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    return data

data = load_data()

# ---------- CACHE MODEL ----------
@st.cache_resource
def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data['review'], data['sentiment'], test_size=0.2, random_state=42
    )
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    return vectorizer, model

vectorizer, model = train_model(data)

# ---------- USER INPUT ----------
user_input = st.text_area("Enter a movie review:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review!")
    else:
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)[0]
        if prediction == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😡")