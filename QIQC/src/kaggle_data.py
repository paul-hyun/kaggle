import codecs

import numpy as np
import pandas as pd

from keras import backend as K
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

#
# Defind
#
DATA_DIR = "../input/"
NROWS = None  # read count: None = all

N_EPOCH = 10
N_BATCH = 1000 if NROWS is None else NROWS // 3

N_EMBED = 300
MAX_FEATURES = 300000
MAXLEN = 70

#
# load data from train.csv, test.csv
#
def load_data():
    # load data
    train_df = pd.read_csv(DATA_DIR + "train.csv", nrows=NROWS)
    test_df = pd.read_csv(DATA_DIR + "test.csv", nrows=None)

    # fill up the values
    train_X = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    # Tokenize the sentences
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    # Pad the sentences
    train_X = pad_sequences(train_X, maxlen=MAXLEN)
    test_X = pad_sequences(test_X, maxlen=MAXLEN)

    # Get the target values
    train_y = train_df['target'].values
    test_id = test_df["qid"].values

    return train_X, test_X, train_y, tokenizer.word_index, test_id

#
# load embedding matrix
#
def load_embedding(word_index, EMBEDDING_FILE):
    nb_words = len(word_index) + 1
    embedding_matrix = [None] * nb_words

    for word, index in word_index.items():
        embedding_matrix[index] = word

    if NROWS == None:
        def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
        with open(EMBEDDING_FILE, encoding='UTF8') as f:
            for o in f:
                if len(o) > 100:
                    word, embedding_vector = get_coefs(*o.split(" "))
                    if word in word_index:
                        index = word_index[word]
                        embedding_matrix[index] = embedding_vector

    return embedding_matrix

#
# check is sring ascii
#
def is_ascii(s):
    return all(ord(c) < 128 for c in s)

#
# main task
#
train_X, test_X, train_y, word_index, test_id = load_data()
print("Train length : ", len(train_X))
print("Test length : ", len(test_X))
# embedding_matrix = load_embedding(word_index, DATA_DIR + "embeddings/glove.840B.300d/glove.840B.300d.txt")
# embedding_matrix = load_embedding(word_index, DATA_DIR + "embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec")
embedding_matrix = load_embedding(word_index, DATA_DIR + "embeddings/paragram_300_sl999/paragram_300_sl999.txt")

un_embeddings = []
for matrix in embedding_matrix:
    if isinstance(matrix, str):
        # print(matrix)
        # pass
        un_embeddings.append(matrix)
    else:
        # print(matrix)
        pass

with codecs.open("paragram_out.txt", 'w', "utf-8") as f:
    for un_embedding in un_embeddings:
        f.write(un_embedding)
        f.write("\n")
