# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import gensim
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from keras import layers
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

#functions
def cleanText(text):
    cleaned = re.sub("[^a-zA-Z0-9']", " ", text)
    cleaned = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
    lowered = cleaned.lower()

    return lowered.strip()

def depunc(sentence):
    sentence = sentence.lower()
    for char in sentence:
        if char not in "abcdefghijklmnopqrstuvwxyz' ":
            sentence = sentence.replace(char,'')
    return sentence

def extract_text(x, y):
    y_out = []
    x_out = []
    wset = set()
    for i in range(len(y)):
        r = x[i]
        r = depunc(r)
        r = r.split()
        wset.update(r)
        x_out.append(r)
        y_out.append(y[i])
    return x_out, y_out, wset


def vectorize(words, indexing):
    all_matrix = []
    for r in words:
        matrix = [0] * len(indexing)
        for w in r:
            if w in indexing:
                matrix[indexing[w]] += 1
        all_matrix.append(matrix)
    return all_matrix

#prepreprocessing
with open("./data/data.csv", "r+", errors = 'ignore') as f:
    df = pd.read_csv(f, delimiter = ",")

print(df['gender'].isnull().sum(), "NaN columns dropped.")
df = df.dropna()

X, y = np.asarray(df["text"]), np.asarray(df["gender"])
X, y, wset = extract_text(X, y)
indexing = {word:idx for idx, word in enumerate(wset)}
indexinginv = {idx:word for idx, word in enumerate(wset)}
print("number of examples:",len(X))

#man, what a horribly inefficient function
X = vectorize(X, indexing)
for i in range(len(y)):
    if y[i] == "male":
        y[i] = 0
    elif y[i] == "female":
        y[i] = 1
    else:
        y[i] = 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

rf = RandomForestClassifier(verbose = True)
print("training...")
rf.fit(X_train, y_train)  
print("predict!")
y_pred = rf.predict(X_test)
#cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

with open("./models/model_rfc", "wb") as f:
    pickle.dump(rf, f)