from flask import Flask, request, render_template
import re
import spacy
import numpy as np
from gensim.models import Word2Vec
import pickle

app = Flask(__name__)

# Open the file in binary read mode
with open("svc_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Load the spaCy model and Word2Vec model
nlp = spacy.load("en_core_web_sm")
word_model= Word2Vec.load("word2vec_model.bin")  # Adjust the path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        news = request.form['news']
        
        # Preprocess the news text
        review = re.sub(r'[^a-zA-Z\s]', '', news)  # Remove non-alphabetic characters
        review = review.lower()  # Convert to lowercase

        # Tokenize and lemmatize using spaCy
        doc = nlp(review)
        corpus = [token.lemma_ for token in doc if not token.is_stop]

        # Convert to Word2Vec input
        vectorized_input_data = [word_model.wv[word] for word in corpus if word in word_model.wv]

        if not vectorized_input_data:
            prediction = "No valid words to vectorize."
        else:
            vectorized_input_data = np.mean(vectorized_input_data, axis=0).reshape(1, -1)
            # Make prediction (assuming `loaded_model` is defined and loaded)
            prediction = loaded_model.predict(vectorized_input_data)
            prediction = "Fake⚠ News Alert🚨📰" if prediction[0] == 0 else "Real News📰"

    return render_template('index.html', prediction=prediction)

@app.route('/clear', methods=['GET'])
def clear():
    return render_template('index.html', prediction=None, news='')

if __name__ == '__main__':
    app.run(port=5001,debug=True)
