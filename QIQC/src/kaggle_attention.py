import numpy as np
import pandas as pd

from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.layers import Dense, Embedding, Input, Dropout
from keras.layers import Bidirectional, LSTM, Layer, Conv1D, GlobalMaxPooling1D, concatenate
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#
# Defind
#
DATA_DIR = "../input/"
NROWS = 10  # read count: None = all

N_EPOCH = 3
N_BATCH = 1000 if NROWS is None else NROWS // 3

N_EMBED = 300
MAX_FEATURES = 300000
MAXLEN = 70
THRESHOLD = 0.34

#
# load data from train.csv, test.csv
#
def load_data():
    # load data
    train_df = pd.read_csv(DATA_DIR + "train.csv", nrows=NROWS)
    test_df = pd.read_csv(DATA_DIR + "test.csv", nrows=NROWS)

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
    print(train_X)
    test_X = pad_sequences(test_X, maxlen=MAXLEN)

    # Get the target values
    train_y = train_df['target'].values
    test_id = test_df["qid"].values

    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.05, random_state=42, stratify=train_y)
    return train_X, valid_X, test_X, train_y, valid_y, tokenizer.word_index, test_id


#
# Attention
#
# https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


#
# build model
#
def build_model(vocab_size):
    inp = Input(shape=(MAXLEN,), dtype='float64')
    input_embedding = Embedding(input_dim=vocab_size+1, output_dim=128, input_length=MAXLEN)(inp)
    
    drop1 = Dropout(0.2)(input_embedding)
    lstm1 = Bidirectional(LSTM(128, return_sequences=True))(drop1)
    conv1 = Conv1D(128, 3, padding='same', activation='relu')(drop1)
    atten1 = Attention(MAXLEN)(lstm1)
    atten2 = Attention(MAXLEN)(conv1)
    max_pool = GlobalMaxPooling1D()(drop1)
    conc1 = concatenate([atten1, atten2, max_pool])
    dens1 = Dense(units=256, activation='relu')(conc1)
    drop2 = Dropout(rate=0.2)(dens1)
    oup = Dense(1, activation='sigmoid')(drop2)

    model = Model(inputs=inp, outputs=oup)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
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
print('='*60)
for epoch in range(N_EPOCH):
    model.fit(train_X, train_y, epochs=1, batch_size=N_BATCH, verbose=0)
    pred_Y = model.predict(valid_X)
    pred_Y = pred_Y.reshape(-1) >= THRESHOLD
    pred_Y = pred_Y.astype(int)
    print(epoch, ':', f1_score(valid_y, pred_Y, average='macro'))
print('='*60)

# predict
prediction = model.predict(test_X)

# make submission
prediction = prediction.reshape(-1)
output = pd.DataFrame(data={"qid": test_id, "prediction": prediction >= THRESHOLD})
output.to_csv("submission.csv", index=False, quoting=3)
