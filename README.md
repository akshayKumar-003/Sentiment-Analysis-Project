# Sentiment Analysis using Machine Learning

## Project overview
Classifies English-language reviews as **positive**, **negative**, or **neutral** using text cleaning, TF-IDF features, and Logistic Regression. A Streamlit app provides an interactive interface.

## Dataset
- File: [`dataset.csv`](dataset.csv)
- Columns: `review`, `sentiment`
- Size: 30 rows; 10 examples per class.
- Dataset source URL: **Not supplied**. The included CSV is the demo dataset. If your internship requires a public dataset citation, replace or supplement it with a properly licensed dataset and add its source URL here.

> **Limitation:** 30 rows is not enough for reliable generalization. The six-row holdout test is only a demonstration.

## Technologies
Python, Pandas, scikit-learn (TF-IDF and Logistic Regression), joblib, Matplotlib, Seaborn, Jupyter Notebook, Streamlit.

## Files
- `YourName_SentimentAnalysis.ipynb` — data exploration, preprocessing, training, evaluation, and model saving.
- `train_model.py` — trains/evaluates model and saves artifacts.
- `app.py` — Streamlit prediction interface.
- `dataset.csv` — labeled sample reviews.
- `model.pkl`, `vectorizer.pkl` — serialized model and TF-IDF vectorizer.
- `requirements.txt` — dependencies.
- `YourName_ProjectReport.docx` — project report; fill in your details.

## Setup and run
Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate:
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate:

```bash
python train_model.py
```

Launch the web app:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit. Enter an English review and click **Analyze Sentiment**. To use the notebook, open `YourName_SentimentAnalysis.ipynb` in Jupyter and run cells from top to bottom.

## Method
1. Load and validate the CSV.
2. Clean text (lowercase, remove URLs and non-letter characters, remove common English stopwords).
3. Split data into training and test sets with stratification.
4. Transform text with TF-IDF unigrams and bigrams.
5. Train Logistic Regression.
6. Evaluate using accuracy, classification report, and confusion matrix.
7. Save model and vectorizer for the Streamlit app.

## Notes
- Run `train_model.py` before launching the app if you want to regenerate model artifacts.
- Training and app prediction use the same text-cleaning logic.
- Model probability is an estimate, not a guarantee of correctness.
- Replace `YourName` in the notebook/report filenames with your actual name before submission.
