import pandas as pd
import numpy as np
import gensim
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sequencer import Sequencer

def cleanText(text):
    cleaned = re.sub("[^a-zA-Z0-9']"," ", text)
    lowered = cleaned.lower()
    
    return lowered.strip()

with open("data.csv", "r+", errors = 'ignore') as f:
    df = pd.read_csv(f, delimiter = ",")

print(df['gender'].isnull().sum(), "NaN columns dropped.")
df = df.dropna()

X, y = np.asarray(df["text"]), np.asarray(df["gender"])
X_clean = [cleanText(t) for t in X]
label_map = {gender:index for index, gender in enumerate(np.unique(y))}
y_prep = np.asarray([label_map[l] for l in y])

X_tokenized = [[w for w in sentence.split(" ") if w != ""] for sentence in X_clean]
model = gensim.models.Word2Vec(X_tokenized, vector_size=100)

print("sequencing")
sequencer = Sequencer(all_words = [token for seq in X_tokenized for token in seq],
              max_words = 1200,
              seq_len = 15,
              embedding_matrix = model.wv)

X_vecs = np.asarray([sequencer.textToVector(" ".join(seq)) for seq in X_tokenized])
X_train, X_test, y_train, y_test = train_test_split(X_vecs, y_prep, test_size = 0.3, random_state = 42)

print("training...")
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
print(svm_classifier.score(X_test, y_test))

with open("model_seq", "wb") as f:
    pickle.dump(svm_classifier, f)