import numpy as np
import pandas as pd

from keras import backend as K
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten
from keras.models import Sequential, Model
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
def load_embedding(word_index, EMBEDDING_FILE, emb_mean, emb_std):
    n_words = min(MAX_FEATURES, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (n_words, N_EMBED))

    if NROWS is None:
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
# calculate f1 score
#
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


#
# build model
#
def build_model(embedding_matrix):
    inp = Input(shape=(MAXLEN,))

    input_embedding = Embedding(embedding_matrix.shape[0], N_EMBED, weights=[embedding_matrix], trainable=False)(inp)
    conv1 = Conv1D(128, 5, activation='relu')(input_embedding)
    pool1 = MaxPooling1D(5)(conv1)
    conv2 = Conv1D(128, 5, activation='relu')(pool1)
    pool2 = MaxPooling1D(5)(conv2)
    # conv3 = Conv1D(128, 5, activation='relu')(pool2)
    # pool3 = MaxPooling1D(35)(conv3)  # global max pooling
    flat = Flatten()(pool2)
    dense1 = Dense(128, activation='relu')(flat)
    oup = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=inp, outputs=oup)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


#
# main task
#
train_X, test_X, train_y, word_index, test_id = load_data()
print("Train length : ", len(train_X))
print("Test length : ", len(test_X))
embedding_matrix = load_embedding(word_index, DATA_DIR + "embeddings/glove.840B.300d/glove.840B.300d.txt", -0.005838499, 0.48782197)
# embedding_matrix = load_embedding(word_index, DATA_DIR + "embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec", -0.0033469985, 0.109855495)
# embedding_matrix = load_embedding(word_index, DATA_DIR + "embeddings/paragram_300_sl999/paragram_300_sl999.txt", -0.0053247833, 0.49346462)
print("Embedding matrix : ", len(word_index))

model = build_model(embedding_matrix)

# train
model.fit(train_X, train_y, epochs=N_EPOCH, batch_size=N_BATCH)

if NROWS is None:
    # predict
    prediction = model.predict(test_X)

    # make submission
    prediction = prediction.reshape(-1)
    output = pd.DataFrame(data={"qid": test_id, "prediction": prediction >= 0.5})
    output.to_csv("submission.csv", index=False, quoting=3)
