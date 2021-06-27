import datetime
import calendar
import pickle
import pandas as pd
import flask
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
nltk.download('wordnet')

rep = 'MultilabelBina.pkl'
with open(rep, 'rb') as file:
    multilabel_binarizer= pickle.load(file)

rep = 'Vectorizer.pkl'
with open(rep, 'rb') as file:
    vectorizer_X= pickle.load(file)


rep = 'Best_Classifier.pkl'
with open(rep, 'rb') as file:
    clf= pickle.load(file)

rep = 'stopwords.pkl'
with open(rep, 'rb') as file:
    stop_words= pickle.load(file)


def formating_the_doc(doc):
  from gensim.utils import simple_preprocess
  doc = BeautifulSoup(doc).find_all('p')
  doc = [phrase.text for phrase in doc]
  doc = " ".join(doc)
  # Lowering
  doc = doc.lower()
  # Tokenizing
  tokenizer = RegexpTokenizer(r'\w+')
  doc = tokenizer.tokenize(doc)
  # Removing stop words and most frequent ones
  doc = [word for word in simple_preprocess(str(doc)) if word not in stop_words]
  # Lemmatization
  lemmatizer = WordNetLemmatizer()
  doc = [lemmatizer.lemmatize(token) for token in doc]
  return doc

def get_tags_supervised(Title,Question):
  Bow = Title + Question
  X_test1 = " ".join(Bow)
  X_test2 = [X_test1]
  X_test3 = vectorizer_X.transform(X_test2)
  y_pred = clf.predict(X_test3)
  Ltags = multilabel_binarizer.inverse_transform(y_pred)
  return Ltags

# Fonction flask
app = flask.Flask(__name__, template_folder='templates')
app.config["DEBUG"] = False


# Page d'accueuil (index), la fonction post lance le calcul de distance et affiche la page résultat

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))

    if flask.request.method == 'POST':
        Title = str(flask.request.form['titl'])
        Question = str(flask.request.form['Quest'])
        Resultat = str(get_tags_supervised([Title],[Question]))
        return flask.render_template('resultats.html',Tags=Resultat)



# lancement de la fonction

if __name__ == '__main__':
    app.run()