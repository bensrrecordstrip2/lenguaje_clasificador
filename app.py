from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

max_features=5000 #we set maximum number of words to 5000
maxlen=400 #we set maximum sequence length to 400

def clean_text_data(setence):
    global tok
    setence = setence.lower()
    setence = setence.replace('[^\w\s]','')
    sentence = [setence]
    clean_data = tok.texts_to_sequences([setence])
    tex_transform = tf.keras.preprocessing.sequence.pad_sequences(clean_data, maxlen=maxlen)
    return tex_transform

with open('tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)


@app.route("/",methods = ['GET', 'POST'])
def index():
    if request.method == "POST":
        setence = request.form.get("oracion")
        model = load_model('clasificador_lenguaje.h5')
        predictions = model.predict(clean_text_data(setence))
        idiomas = ['Ingles','Frances','italiano','Español']
        index = predictions.argmax()
        result = idiomas[index]
        return render_template("index.html", data = [{"result":result}])
    else:
        return render_template("index.html")
    
    
    
    
    
if __name__=="__main__":
    app.run()