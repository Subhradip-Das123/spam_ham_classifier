import streamlit as st
from utils import load_model, clean_text

# --- Page Setup ---
st.set_page_config(page_title="Spam–Ham Classifier", page_icon="📧", layout="centered")

# --- Title and Description ---
st.title("📧 Spam–Ham Classifier System")
st.write("""
This intelligent NLP system classifies text messages or emails as **Spam** or **Ham (Not Spam)** 
using a trained Machine Learning model.
""")

# --- Load Model and Vectorizer ---
model, vectorizer = load_model()

# --- Text Input ---
msg = st.text_area("✉️ Enter your message here:", height=150)

# --- Classify Button ---
if st.button("Classify Message"):
    if msg.strip():
        cleaned = clean_text(msg)
        vect_msg = vectorizer.transform([cleaned])
        pred = model.predict(vect_msg)[0]
        prob = model.predict_proba(vect_msg)[0][pred]

        if pred == 1:
            st.error(f"🚫 **Spam Detected!** (Confidence: {prob*100:.2f}%)")
        else:
            st.success(f"✅ **Ham Message!** (Confidence: {prob*100:.2f}%)")
    else:
        st.warning("⚠️ Please enter a message before classifying.")

# --- Footer ---
# st.markdown("""
# ---
# Developed with ❤️ using **Streamlit** and **scikit-learn**
# 📘 *Machine Learning Spam–Ham Classifier Project*
# """)
