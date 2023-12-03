import streamlit as st 
import pickle 
import string 
from nltk.corpus import stopwords
import nltk 
import numpy as np

word2vec_model = pickle.load(open('word2vec_model.pkl','rb'))
xgb2 = pickle.load(open('xgb2.pkl', 'rb'))

def transform_text(text):
    text = text.lower() # Lowercasing:Convert all text to lowercase to ensure consistency.
    text = nltk.word_tokenize(text) # Tokenization:Tokenize the text into individual words or phrases.

    y = []
    for i in text :
        if i.isalpha(): # Keeping only alphabetical words.
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation: # Removing Stopwords.
            y.append(i)
    return " ".join(y)

# Average word2vec (function for - one vector for one review)
def document_vector(doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc.split() if word in word2vec_model.wv.index_to_key]
    return np.mean(word2vec_model.wv[doc], axis = 0)

st.title("Book Review Rating Predictor")

input_review = st.text_area("Enter the Review")
if st.button("Predict"):

    # 1. Preprocess
    transformed_review=transform_text(input_review)
    # 2. Vectorize
    t = []
    try:  # Exception handeling, to handel some wierd reviews i.e. hashtags, one word reviews
        t.append(document_vector(transformed_review))
    except:
        t.append(np.random.uniform(-1,1,(100,))) 
        # Adding Randomly generated vector for some troublesome reviews.
    vector_input = np.array(t)
    # 3. Predict
    result = xgb2.predict(vector_input)[0]
    # 4. Display
    st.header(f"Predicted rating is : {result}")