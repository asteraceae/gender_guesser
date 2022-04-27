#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import gensim
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# In[3]:


with open("/Users/evepalmer/Desktop/data.csv", "r+", errors = 'ignore') as f:
    #df = f.readlines()
    df = pd.read_csv(f, delimiter = ",")


# In[16]:


df.head()


# In[6]:


df['gender'].isnull().sum()
# we have a lot of null rows


# In[5]:


df.dropna()


# In[11]:


# do we need to do a better job of cleaning to get rid of photo links/deal with hashtags ?
def cleanText(text):
    cleaned = re.sub("[^a-zA-Z0-9']"," ",text)
    lowered = cleaned.lower()
    
    return lowered.strip()


# In[14]:


X,y = np.asarray(df["text"]),np.asarray(df["gender"])

X_clean = [cleanText(t) for t in X]
X_clean[:4]


# In[15]:


label_map = {cat:index for index,cat in enumerate(np.unique(y))}
y_prep = np.asarray([label_map[l] for l in y])

label_map


# In[17]:


X_tokenized = [[w for w in sentence.split(" ") if w != ""] for sentence in X_clean]
X_tokenized[0]


# In[18]:


# adjust hyperparams?
model = gensim.models.Word2Vec(X_tokenized, vector_size=100)


# In[19]:


# class for text to word embedding sequences
class Sequencer():
    
    def __init__(self,
                 all_words,
                 max_words,
                 seq_len,
                 embedding_matrix
                ):
        
        self.seq_len = seq_len
        self.embed_matrix = embedding_matrix
        """
        temp_vocab = Vocab which has all the unique words
        self.vocab = Our last vocab which has only most used N words.
    
        """
        temp_vocab = list(set(all_words))
        self.vocab = []
        self.word_cnts = {}
        """
        Now we'll create a hash map (dict) which includes words and their occurencies
        """
        for word in temp_vocab:
            # 0 does not have a meaning, you can add the word to the list
            # or something different.
            count = len([0 for w in all_words if w == word])
            self.word_cnts[word] = count
            counts = list(self.word_cnts.values())
            indexes = list(range(len(counts)))
        
        # Now we'll sort counts and while sorting them also will sort indexes.
        # We'll use those indexes to find most used N word.
        cnt = 0
        while cnt + 1 != len(counts):
            cnt = 0
            for i in range(len(counts)-1):
                if counts[i] < counts[i+1]:
                    counts[i+1],counts[i] = counts[i],counts[i+1]
                    indexes[i],indexes[i+1] = indexes[i+1],indexes[i]
                else:
                    cnt += 1
        
        for ind in indexes[:max_words]:
            self.vocab.append(temp_vocab[ind])
                    
    def textToVector(self,text):
        # First we need to split the text into its tokens and learn the length
        # If length is shorter than the max len we'll add some spaces (100D vectors which has only zero values)
        # If it's longer than the max len we'll trim from the end.
        tokens = text.split()
        len_v = len(tokens)-1 if len(tokens) < self.seq_len else self.seq_len-1
        vec = []
        for tok in tokens[:len_v]:
            try:
                vec.append(self.embed_matrix[tok])
            except Exception as E:
                pass
        
        last_pieces = self.seq_len - len(vec)
        for i in range(last_pieces):
            vec.append(np.zeros(100,))
        
        return np.asarray(vec).flatten()      


# In[ ]:


sequencer = Sequencer(all_words = [token for seq in X_tokenized for token in seq],
              max_words = 1200,
              seq_len = 15,
              embedding_matrix = model.wv)


# In[89]:


X_vecs = np.asarray([sequencer.textToVector(" ".join(seq)) for seq in X_tokenized])


# In[93]:


X_train, X_test, y_train, y_test = train_test_split(X_vecs, y_prep, test_size = 0.2, random_state = 42)


# In[94]:


svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)


# In[97]:


svm_classifier.score(X_test, y_test)

