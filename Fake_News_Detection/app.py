import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# App UI
st.title("📰 Fake News Detector")
st.write("Enter a news article below to predict whether it's **Real** or **Fake**.")

# Input text
user_input = st.text_area("Paste your news article text here", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize the input
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        prob = model.predict_proba(input_vector)[0][prediction]

        if prediction == 1:
            st.error(f"🛑 Prediction: **Fake News** (Confidence: {prob:.2%})")
        else:
            st.success(f"✅ Prediction: **Real News** (Confidence: {prob:.2%})")