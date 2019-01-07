import numpy as np
import pandas as pd

from keras import backend as K
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Flatten
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#
# Defind
#
DATA_DIR = "../input/"
NROWS = 1000  # read count: None = all

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

    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=42)
    return train_X, valid_X, test_X, train_y, valid_y, tokenizer.word_index, test_id


#
# build model
#
def build_model(vocab_size):
    inp = Input(shape=(MAXLEN,), dtype='float64')

    input_embedding = Embedding(input_dim=vocab_size+1, output_dim=128,input_length=MAXLEN)(inp)
    drop1 = Dropout(0.2)(input_embedding)
    conv1 = Conv1D(32, 3, activation='relu')(drop1)
    # conv2 = Conv1D(128, 3, activation='relu')(pool1)
    # pool2 = MaxPooling1D(3)(conv2)
    # conv3 = Conv1D(128, 3, activation='relu')(pool2)
    # pool3 = MaxPooling1D(3)(conv3)  # global max pooling
    final = GlobalMaxPooling1D()(conv1)
    # final = Flatten()(conv1)
    dens1 = Dense(units=250, activation='relu')(final)
    drop2 = Dropout(rate=0.2)(dens1)
    oup = Dense(1, activation='sigmoid')(drop2)

    model = Model(inputs=inp, outputs=oup)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


#
# main task
#
train_X, valid_X, test_X, train_y, valid_y, word_index, test_id = load_data()
print("Train length : ", len(train_X))
print("Valid length : ", len(valid_X))
print("Test length : ", len(test_X))
print("Embedding matrix : ", len(word_index))

model = build_model(len(word_index))

# trains
for epoch in range(N_EPOCH):
    model.fit(train_X, train_y, epochs=1, batch_size=N_BATCH, verbose=0)
    pred_Y = model.predict(valid_X)
    pred_Y = pred_Y.reshape(-1) >= 0.5
    pred_Y = pred_Y.astype(int)
    print(epoch, ':', f1_score(valid_y, pred_Y, average='macro'))
    print('='*60)

# predict
prediction = model.predict(test_X)

# make submission
prediction = prediction.reshape(-1)
output = pd.DataFrame(data={"qid": test_id, "prediction": prediction >= 0.5})
output.to_csv("submission.csv", index=False, quoting=3)
