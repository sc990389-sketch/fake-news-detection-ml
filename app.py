import streamlit as st
import pandas as pd
import pickle
import re

# -------------------------------
# Load Model & Vectorizer
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# Text Cleaning Function
# -------------------------------
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Enter any news text below to check whether it is Real or Fake.")

# Input box
news = st.text_area("Enter News Here:")

# Button
if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Preprocess
        cleaned_text = wordopt(news)

        # Vectorize
        vector = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(vector)

        # Output
        if prediction[0] == 0:
            st.error("🚨 Fake News Detected")
        else:
            st.success("✅ Real News")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | AI Project")