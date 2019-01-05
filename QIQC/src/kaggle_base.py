import os
import time
import numpy as np
import pandas as pd

from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense
from keras.models import Model
from keras import backend as K

#
# Defind
#
DATA_DIR = "../input/"
NROWS = 10 # read count

N_EPOCH = 10
N_BATCH = 500 if NROWS is None else NROWS // 3

N_EMBED = 300
MAX_FEATURES = 95000
MAXLEN = 70

def load_and_prec():
    train_df = pd.read_csv(DATA_DIR + "train.csv", nrows=NROWS)
    test_df = pd.read_csv(DATA_DIR + "test.csv", nrows=NROWS)
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=MAXLEN)
    test_X = pad_sequences(test_X, maxlen=MAXLEN)

    ## Get the target values
    train_y = train_df['target'].values
    
    #shuffling the data
    np.random.seed(2018)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]
    
    return train_X, test_X, train_y, tokenizer.word_index


def load_glove(word_index):
    nb_words = min(MAX_FEATURES, len(word_index) + 1)
    # embedding_matrix = np.zeros((nb_words, N_EMBED))
    embedding_matrix = np.random.normal(size=(nb_words, N_EMBED))

    # EMBEDDING_FILE = DATA_DIR + "embeddings/glove.840B.300d/glove.840B.300d.txt"
    # def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    # with open(EMBEDDING_FILE, encoding='UTF8') as f:
    #     for o in f:
    #         if len(o) > 100:
    #             word, embedding_vector = get_coefs(*o.split(" "))
    #             if word in word_index:
    #                 index = word_index[word]
    #                 embedding_matrix[index] = embedding_vector

    return embedding_matrix

def load_fasttext(word_index):    
    nb_words = min(MAX_FEATURES, len(word_index) + 1)
    embedding_matrix = np.zeros((nb_words, N_EMBED))

    EMBEDDING_FILE = DATA_DIR + "embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec"
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    with open(EMBEDDING_FILE, encoding='UTF8') as f:
        for o in f:
            if len(o) > 100:
                word, embedding_vector = get_coefs(*o.split(" "))
                if word in word_index:
                    index = word_index[word]
                    embedding_matrix[index] = embedding_vector

    return embedding_matrix


def load_para(word_index):
    nb_words = min(MAX_FEATURES, len(word_index) + 1)
    embedding_matrix = np.zeros((nb_words, N_EMBED))

    EMBEDDING_FILE = DATA_DIR + "embeddings/paragram_300_sl999/paragram_300_sl999.txt"
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    with open(EMBEDDING_FILE, encoding='UTF8') as f:
        for o in f:
            if len(o) > 100:
                word, embedding_vector = get_coefs(*o.split(" "))
                if word in word_index:
                    index = word_index[word]
                    embedding_matrix[index] = embedding_vector

    return embedding_matrix


def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def train_pred(model, train_X, train_y, val_X, val_y, epochs=2, callback=None):
    for e in range(epochs):
        model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y), callbacks = callback, verbose=0)
        pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)

        best_score = metrics.f1_score(val_y, (pred_val_y > 0.33).astype(int))
        print("Epoch: ", e, "-    Val F1 Score: {:.4f}".format(best_score))

    pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
    print('=' * 60)
    return pred_val_y, pred_test_y, best_score


def build_model(embedding_matrix):
    input = Input(shape=(MAXLEN,))
    input_embedding = Embedding(embedding_matrix.shape[0], N_EMBED, weights=[embedding_matrix], trainable=False)(input)

    output = Dense(1, activation="sigmoid")(input_embedding)

    model = Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score])
    
    return model


train_X, test_X, train_y, word_index = load_and_prec()
embedding_matrix = load_glove(word_index)
# embedding_matrix = load_fasttext(word_index)
# embedding_matrix = load_para(word_index)
print(len(word_index))
print(embedding_matrix.shape)


# DATA_SPLIT_SEED = 2018
# clr = CyclicLR(base_lr=0.001, max_lr=0.002,
#                step_size=300., mode='exp_range',
#                gamma=0.99994)

# train_meta = np.zeros(train_y.shape)
# test_meta = np.zeros(test_X.shape[0])
# splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=DATA_SPLIT_SEED).split(train_X, train_y))
# for idx, (train_idx, valid_idx) in enumerate(splits):
        # X_train = train_X[train_idx]
        # y_train = train_y[train_idx]
        # X_val = train_X[valid_idx]
        # y_val = train_y[valid_idx]
        # model = model_lstm_atten(embedding_matrix)
        # pred_val_y, pred_test_y, best_score = train_pred(model, X_train, y_train, X_val, y_val, epochs = 8, callback = [clr,])
        # train_meta[valid_idx] = pred_val_y.reshape(-1)
        # test_meta += pred_test_y.reshape(-1) / len(splits)

# model = build_model(embedding_matrix)
# pred_val_y, pred_test_y, best_score = train_pred(model, train_X, y_train, X_val, y_val, epochs = 8, callback = [clr,])

# sub = pd.read_csv('../input/sample_submission.csv')
# sub.prediction = test_meta > 0.33
# sub.to_csv("submission.csv", index=False)
# f1_score(y_true=train_y, y_pred=train_meta > 0.33)