import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------- TITLE -----------------
st.title("🎬 Movie Review Sentiment Analysis")

# ----------------- LOAD DATA -----------------
@st.cache_data
def load_data():
    url = "https://github.com/Ather17-mughal/AI_project/releases/download/csv/IMDB.Dataset.csv"
    
    # Load only 5000 rows for speed
    df = pd.read_csv(url).sample(5000, random_state=42)

    df['review'] = df['review'].str.lower()
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    return df

data = load_data()

st.write(f"✅ Dataset loaded: {data.shape[0]} rows")

# ----------------- TRAIN MODEL -----------------
@st.cache_resource
def train_model(data):

    X_train, X_test, y_train, y_test = train_test_split(
        data['review'],
        data['sentiment'],
        test_size=0.2,
        random_state=42
    )

    # Limit features for speed
    vectorizer = CountVectorizer(max_features=5000)

    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=200)

    model.fit(X_train_vec, y_train)

    accuracy = model.score(
        vectorizer.transform(X_test),
        y_test
    )

    return vectorizer, model, accuracy

vectorizer, model, accuracy = train_model(data)

st.write(f"🎯 Accuracy: {accuracy*100:.2f}%")

# ----------------- USER INPUT -----------------
user_input = st.text_area("Enter a movie review:")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review!")

    else:
        user_vec = vectorizer.transform([user_input.lower()])

        prediction = model.predict(user_vec)[0]

        if prediction == 1:
            st.success("😊 Positive Review")

        else:
            st.error("😡 Negative Review")
