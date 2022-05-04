# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gensim
import re
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.models import Sequential
from keras import layers
from keras import backend as K
from keras.callbacks import ModelCheckpoint


#import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
print(tf. __version__)

def depunc(sentence):
    sentence = sentence.lower()
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    sentence = url_pattern.sub(r'', sentence)
    sentence = re.sub("\'", "", sentence)
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
        temp = r
        r = r.split()
        wset.update(r)
        x_out.append(temp)
        y_out.append(y[i])
    return x_out, y_out, wset

with open("./data.csv", "r+", errors = 'ignore') as f:
    df = pd.read_csv(f, delimiter = ",")

print(df['gender'].isnull().sum(), "NaN columns dropped.")
df = df.dropna()
X, y = np.asarray(df["text"]), np.asarray(df["gender"])

X, y, wset = extract_text(X, y)

indexing = {word:idx for idx, word in enumerate(wset)}
indexinginv = {idx:word for idx, word in enumerate(wset)}

for i in range(len(y)):
    if y[i] == "male":
        y[i] = 0
    elif y[i] == "female":
        y[i] = 1
    else:
        y[i] = 2
        
print("number of examples:",len(X))

max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
tweets = pad_sequences(sequences, maxlen = max_len)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(tweets, y, test_size = 0.30, random_state = 0)
print(X_train)

rf = RandomForestClassifier(verbose = True)
print("training...")
rf.fit(X_train, y_train)  
print("predict!")
y_pred = rf.predict(X_test)
#cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

embedding_layer = Embedding(1000, 3)
model1 = Sequential()
model1.add(layers.Embedding(max_words, 20)) #The embedding layer
model1.add(layers.LSTM(15,dropout=0.5)) #Our LSTM layer
model1.add(layers.Dense(3,activation='softmax'))


model1.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint1 = ModelCheckpoint("best_model1.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)
history = model1.fit(X_train, y_train, epochs=70,validation_data=(X_test, y_test),callbacks=[checkpoint1])

