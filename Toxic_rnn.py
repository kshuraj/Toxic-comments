# -*- coding: utf-8 -*-


import numpy as np # linear algebra
import pandas as pd # data processing
import os

os.chdir('F:\Kaggle Competitions\Toxic')

#########
#Import packages
im port sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


#Import data set
EMBEDDING_FILE='glove6b100dtxt/glove.6B.100d.txt'
TRAIN_DATA_FILE='train.csv\\train.csv'
TEST_DATA_FILE='test.csv\\test.csv'


#####set parameters
embed_size = 100 # word vector size
max_features = 25000 # unique words to use (i.e num rows in embedding vector)
maxlen = 100

#read datasets
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

#Stemming and lematisation
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer

corpus = []
for i in range(0, 159571):
    Conversation = train.comment_text.astype(str)
    Conversation = re.sub('[^a-zA-Z]', ' ', Conversation[i])
    Conversation = Conversation.lower()
    Conversation = Conversation.split()
    Conversation = [word for word in Conversation if not word in set(stopwords.words('english'))]
    ps = LancasterStemmer()
    Conversation = [ps.stem(word) for word in Conversation if not word in set(stopwords.words('english'))]
    Conversation = ' '.join(Conversation)
    corpus.append(Conversation)  
    
corpus1 = []
for i in range(0, 153164):
    Conversation = test.comment_text.astype(str)
    Conversation = re.sub('[^a-zA-Z]', ' ', Conversation[i])
    Conversation = Conversation.lower()
    Conversation = Conversation.split()
    Conversation = [word for word in Conversation if not word in set(stopwords.words('english'))]
    ps = LancasterStemmer()
    Conversation = [ps.stem(word) for word in Conversation if not word in set(stopwords.words('english'))]
    Conversation = ' '.join(Conversation)
    corpus1.append(Conversation) 

train["comment_text"] = corpus
test["comment_text"] = corpus1


####Pre-processing
list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_tr = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

###
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_tr1 = sc.fit_transform(X_tr)
X_te1 = sc.transform(X_te)


def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE,encoding="utf8"))


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(10, activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
    
model.fit(X_tr, y, batch_size=32, epochs=10, validation_split=0.1);
    
y_test = model.predict([X_te], batch_size=1024, verbose=1) 
    
sample_submission = pd.read_csv('sample_submission.csv\\sample_submission.csv') 
sample_submission[list_classes] = y_test
sample_submission.to_csv('submission.csv', index=False)

















