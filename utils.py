import pickle
import re

# --- Load the model and vectorizer ---
def load_model():
    try:
        model = pickle.load(open("models/spam_model.pkl", "rb"))
        vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
        return model, vectorizer
    except FileNotFoundError:
        raise Exception("❌ Model or Vectorizer file not found in 'models/' folder. Please check paths.")

# --- Text cleaning helper ---
def clean_text(text):
    # Lowercase and remove punctuation/numbers
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
