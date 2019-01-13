import re
import numpy as np
import pandas as pd

from keras import optimizers
from keras import  backend as K, initializers, regularizers, constraints
from keras.layers import Dense, Embedding, Input, Dropout, SpatialDropout1D
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten
from keras.layers import Bidirectional, LSTM
from keras.layers import Layer, concatenate
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
NROWS = 1000  # read count: None = all

N_EPOCH = 100
N_BATCH = 1000 if NROWS is None else NROWS // 3

N_EMBED = 300
TRAIN_EMBED = True
MAX_FEATURES = 100000
MAXLEN = 70
LEARN_RATE = 0.0001

#
# Clean Text
#
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
 
def clean_text(x):
    x = str(x) 
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4,}', '####', x)
    x = re.sub('[0-9]{3,}', '###', x)
    x = re.sub('[0-9]{2,}', '##', x)
    return x


#
# miss spell
#
mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",
"couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
"haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
"how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
"I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have",
"i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", 
"it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
"mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
"needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
"shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
"she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",
"shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would",
"that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have",
"there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will",
"they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
"we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
"we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
"what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
"where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
"who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
"would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
"y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have",
"you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center',
'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled',
'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora',
'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does',
'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum',
'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization',
'demonetisation': 'demonetization'}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)


#
# load data from train.csv, test.csv
#
def load_data():
    # load data
    train_df = pd.read_csv(DATA_DIR + "train.csv", nrows=NROWS)
    test_df = pd.read_csv(DATA_DIR + "test.csv", nrows=NROWS)

    for df in [train_df, test_df]:
        df["question_text"] = df["question_text"].str.lower()
        df["question_text"] = df["question_text"].apply(lambda x: clean_text(x))
        df["question_text"] = df["question_text"].apply(lambda x: clean_numbers(x))
        df["question_text"] = df["question_text"].apply(lambda x: replace_typical_misspell(x))
        df["question_text"].fillna("_##_", inplace=True)

    # fill up the values
    train_X = train_df["question_text"].values
    test_X = test_df["question_text"].values

    # Tokenize the sentences
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    all_Text = []
    all_Text.extend(train_X)
    # all_Text.extend(test_X)
    tokenizer.fit_on_texts(list(all_Text))
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
# embedding
#
def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(MAX_FEATURES, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 
    
def load_fasttext(word_index):    
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(MAX_FEATURES, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(MAX_FEATURES, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


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
# build cnn model
#
def build_cnn(embedding_matrix):
    inp = Input(shape=(MAXLEN,), dtype='float64')
    input_embedding = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=MAXLEN, trainable=TRAIN_EMBED)(inp)
    
    drop1 = SpatialDropout1D(0.2)(input_embedding)
    conv1 = Conv1D(128, 3, activation='relu')(drop1)
    pool1 = GlobalMaxPooling1D()(conv1)
    dens1 = Dense(units=256, activation='relu')(pool1)
    drop2 = Dropout(rate=0.2)(dens1)
    oup = Dense(1, activation='sigmoid')(drop2)

    model = Model(inputs=inp, outputs=oup)
    adam = optimizers.Adam(lr=LEARN_RATE)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

#
# build lstm model
#
def build_lstm(embedding_matrix):
    inp = Input(shape=(MAXLEN,), dtype='float64')
    input_embedding = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=MAXLEN, trainable=TRAIN_EMBED)(inp)
    
    drop1 = SpatialDropout1D(0.2)(input_embedding)
    lstm1 = Bidirectional(LSTM(128))(drop1)
    dens1 = Dense(units=256, activation='relu')(lstm1)
    drop2 = Dropout(rate=0.2)(dens1)
    oup = Dense(1, activation='sigmoid')(drop2)

    model = Model(inputs=inp, outputs=oup)
    adam = optimizers.Adam(lr=LEARN_RATE)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


#
# build attention model
#
def build_attention(embedding_matrix):
    inp = Input(shape=(MAXLEN,), dtype='float64')
    input_embedding = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=MAXLEN, trainable=TRAIN_EMBED)(inp)
    
    drop1 = SpatialDropout1D(0.2)(input_embedding)
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
    adam = optimizers.Adam(lr=LEARN_RATE)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


#
# main task
#
train_X, test_X, train_y, word_index, test_id = load_data()
print("Train length : ", len(train_X))
print("Test length : ", len(test_X))
# embedding_matrix1 = load_glove(word_index)
# print("Embedding glove : ", embedding_matrix1.shape)
# embedding_matrix2 = load_fasttext(word_index)
# print("Embedding fasttext : ", embedding_matrix2.shape)
# embedding_matrix3 = load_para(word_index)
# print("Embedding para : ", embedding_matrix3.shape)
# embedding_matrix = np.mean([embedding_matrix1, embedding_matrix2, embedding_matrix3], axis = 0)
# embedding_matrix = np.concatenate((embedding_matrix1, embedding_matrix2, embedding_matrix3), axis=1)
embedding_matrix = np.random.normal(0, 0.01, (min(MAX_FEATURES, len(word_index) + 1), N_EMBED))
print("Embedding matrix : ", embedding_matrix.shape)

models = []
models.append({'name': 'cnn', 'model':build_cnn(embedding_matrix), 'epoch': 10})
models.append({'name': 'rnn', 'model':build_lstm(embedding_matrix), 'epoch': 10})
# models.append({'name': 'attention', 'model':build_attention(embedding_matrix), 'epoch': 10})


#
# train
#
def train(model, epochs, train_X, train_y):
    max_epoch = 0
    max_f1 = 0.0
    max_th = 0.0
    max_prediction = None

    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.05, random_state=42, stratify=train_y)
    for epoch in range(epochs):
        model.fit(train_X, train_y, epochs=1, batch_size=N_BATCH, verbose=0)
        pred_Y = model.predict(valid_X)
        f1_val = 0.0
        th_val = 0.0
        for th in np.arange(0.25, 0.75, 0.01):
            f1 = metrics.f1_score(valid_y, (pred_Y > th).astype(int))
            if f1_val < f1:
                f1_val, th_val = f1, th
        print('\tEpoch: {:2d}, F1: {:.4f}, Threshold: {:.4f}'.format(epoch, f1_val, th_val))
        if max_f1 < f1_val:
            max_epoch = epoch
            max_f1 = f1_val
            max_th = th_val
            max_prediction = (model.predict(test_X).reshape(-1) > th_val).astype(int)
            print('\t\tMax f1, make prediction')
    return max_epoch, max_f1, max_th, max_prediction


max_values = []
for item in models:
    print('='*60)
    print(item)
    epoch, f1, threshold, prediction = train(item['model'], item['epoch'], train_X, train_y)
    max_values.append({'item': item, 'epoch': epoch, 'f1': f1, 'threshold': threshold, 'prediction': prediction})
    print('='*60)


#
# submission max prediction
#
def submission_max(test_id, max_values):
    max_value = max_values[0]
    for vvv in max_values:
        if max_value['f1'] < vvv['f1']:
            max_value = vvv
    print('Save submission.csv Name: {}, Epoch: {:2d}, F1: {:.4f}, Threshold: {:.4f}'.format(max_value['item']['name'], max_value['epoch'], max_value['f1'], max_value['threshold']))
    output = pd.DataFrame(data={"qid": test_id, "prediction": max_value['prediction'] > 0.5})
    output.to_csv("submission.csv", index=False, quoting=3)


submission_max(test_id, max_values)

