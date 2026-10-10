import re
import joblib
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

@st.cache_resource
def load_artifacts():
    return joblib.load("model.pkl"), joblib.load("vectorizer.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return " ".join(word for word in text.split() if word not in ENGLISH_STOP_WORDS)

st.set_page_config(page_title="Sentiment Analysis", page_icon="😊", layout="centered")
st.title("😊 Sentiment Analysis")
st.write("Enter an English review and the trained model will predict its sentiment.")
st.caption("Educational prototype: results depend on the size and quality of the training dataset.")
try:
    model, vectorizer = load_artifacts()
except Exception as exc:
    st.error("Model files could not be loaded. Run `python train_model.py` first.")
    st.stop()

review = st.text_area("Enter your review:", placeholder="Example: I really loved this product!")
if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review)
        if not cleaned.strip():
            st.warning("The review has no usable English words after cleaning. Try a longer review.")
        else:
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            probs = model.predict_proba(vector)[0]
            confidence = float(max(probs) * 100)
            if prediction == "positive":
                st.success("😊 Positive Sentiment")
            elif prediction == "negative":
                st.error("😞 Negative Sentiment")
            else:
                st.info("😐 Neutral Sentiment")
            st.write(f"**Model confidence:** {confidence:.2f}%")
            st.caption("Confidence is the model's estimated class probability, not a guarantee of correctness.")
