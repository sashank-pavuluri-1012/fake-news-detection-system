import streamlit as st
import joblib

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News Detection", page_icon="📰")

st.title("AI-Based Fake News Detection System")

st.write("Enter a news headline or article to check whether it is Real or Fake.")

news = st.text_area("Enter News Here:")

if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter some news first.")
    else:
        news_vector = vectorizer.transform([news])
        prediction = model.predict(news_vector)

        if prediction[0] == 1:
            st.success("Prediction: REAL NEWS")
        else:
            st.error("Prediction: FAKE NEWS")

st.caption("Note: This is an educational ML model, not a professional fact-checking tool.")
