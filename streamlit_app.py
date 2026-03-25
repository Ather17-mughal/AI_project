import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ----------------- LOAD DATA -----------------
url = "https://github.com/Ather17-mughal/AI_project/releases/download/csv/IMDB.Dataset.csv"
data = pd.read_csv(url)

# Convert text to lowercase
data['review'] = data['review'].str.lower()

# Convert labels to numbers
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    data['review'], data['sentiment'], test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ----------------- VECTORIZE TEXT -----------------
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------- TRAIN MODEL -----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ----------------- PREDICTION & EVALUATION -----------------
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))