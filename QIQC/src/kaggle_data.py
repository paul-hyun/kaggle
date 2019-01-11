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
# Clean Text
#
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


#
# load data from train.csv, test.csv
#
def load_data():
    # load data
    train_df = pd.read_csv(DATA_DIR + "train.csv", nrows=NROWS)
    test_df = pd.read_csv(DATA_DIR + "test.csv", nrows=None)

    for df in [train_df, test_df]:
        df["question_text"] = df["question_text"].str.lower()
        df["question_text"] = df["question_text"].apply(lambda x: clean_text(x))
        df["question_text"].fillna("_##_", inplace=True)

    # fill up the values
    train_Text = train_df["question_text"].values
    test_Text = test_df["question_text"].values

    # Tokenize the sentences
    tokenizer = Tokenizer(oov_token='__UNK__')
    all_Text = []
    all_Text.extend(train_Text)
    all_Text.extend(test_Text)
    tokenizer.fit_on_texts(list(all_Text))
    train_X = tokenizer.texts_to_sequences(train_Text)
    test_X = tokenizer.texts_to_sequences(test_Text)

    __UNK__ = tokenizer.word_index['__UNK__']
    count_all = 0
    count_oov = 0
    line_all = 0
    line_oov = 0
    for test_line in test_X:
        flag_oov = False
        for token in test_line:
            count_all += 1
            if token == __UNK__:
                count_oov += 1
                flag_oov = True
        line_all += 1
        if flag_oov:
            line_oov += 1
    print(count_oov, '/', count_all)
    print(line_oov, '/', line_all)

    # Pad the sentences
    train_X = pad_sequences(train_X, maxlen=MAXLEN)
    test_X = pad_sequences(test_X, maxlen=MAXLEN)

    # Get the target values
    train_y = train_df['target'].values
    test_id = test_df["qid"].values

    return train_X, test_X, train_y, tokenizer.word_index, test_id, train_Text, test_Text

#
# check is sring ascii
#
def is_ascii(s):
    return all(ord(c) < 128 for c in s)

#
# main task
#
train_X, test_X, train_y, word_index, test_id, train_Text, test_Text = load_data()
print("Train length : ", len(train_X))
print("Test length : ", len(test_X))
print("Dic length : ", len(word_index))


