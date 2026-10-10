"""Train and evaluate the Sentiment Analysis model."""
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "dataset.csv"

def clean_text(text):
    """Apply the same preprocessing during training and prediction."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return " ".join(word for word in text.split() if word not in ENGLISH_STOP_WORDS)

def main():
    data = pd.read_csv(DATA_PATH)
    required = {"review", "sentiment"}
    if not required.issubset(data.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")
    data = data.dropna(subset=["review", "sentiment"]).copy()
    data["review"] = data["review"].astype(str)
    data["sentiment"] = data["sentiment"].astype(str).str.lower().str.strip()
    data["clean_review"] = data["review"].apply(clean_text)
    data = data[data["clean_review"].str.strip().ne("")]
    X, y = data["clean_review"], data["sentiment"]
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("Need at least two sentiment classes and at least two rows per class.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    print(f"Rows used: {len(data)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix (labels:", list(model.classes_), ")")
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Saved model.pkl and vectorizer.pkl")

if __name__ == "__main__":
    main()
