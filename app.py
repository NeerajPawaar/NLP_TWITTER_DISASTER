import streamlit as st
import pickle
import re
import string

model = pickle.load(open(r"D:\Project-7\disaster_model.pkl","rb"))
vectorizer = pickle.load(open(r"D:\Project-7\tfidf_vectorizer.pkl","rb"))

def clean_text(text):

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text

st.title("Disaster Tweet Classifier")

tweet = st.text_area("Enter a Tweet")

if st.button("Predict"):

    cleaned = clean_text(tweet)

    vect = vectorizer.transform([cleaned])

    prediction = model.predict(vect)[0]

    if prediction == 1:
        st.error("This Tweet is related to a Disaster")
    else:
        st.success("This Tweet is NOT related to a Disaster")