
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.write("Check whether a news headline is likely **Real** or **Fake** using a simple ML model.")

# Small demo dataset
texts = [
    "Government launches new healthcare program",
    "Scientists discover water on Mars",
    "Celebrity adopts new puppy",
    "Aliens landed in my backyard yesterday",
    "Miracle cure for all diseases found overnight",
    "Secret society controlling the world revealed"
]

labels = [1,1,1,0,0,0]  # 1 = Real, 0 = Fake

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

news_input = st.text_area("Enter a news headline or short article")

if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        transformed = vectorizer.transform([news_input])
        prediction = model.predict(transformed)[0]

        if prediction == 1:
            st.success("This news appears to be REAL.")
        else:
            st.error("This news appears to be FAKE.")
