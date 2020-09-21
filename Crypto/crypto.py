import numpy as np
import pandas as pd
import os
import random
from collections import deque
from IPython.display import clear_output

import tensorflow as tf
from sklearn import preprocessing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.python.keras.optimizers import Adam

clear_output()

DATADIR = "C:/Users/User/Desktop/yemal/_data/crypto_data"

SEQUENCE_LEN = 50  # Number of previous data points to use in RNN
FUTURE_LEN = 5  # Number of future data points to predict

# =========================================================

# Reading the data
# ---------------------------------------------------------

dataset = pd.read_csv( os.path.join(DATADIR, "ETH-USD.csv"), names=['time', 'low', 'high', 'open', 'close', 'volume'] )
dataset.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
dataset.dropna(inplace=True)
dataset.set_index("time", inplace=True)

dataset['future'] = dataset['close'].shift(-FUTURE_LEN)


def increaseOrDecrease(past, future):
    # Figure out whether the value increses or decreases
    if float(past) < float(future):
        return 1  # increase
    else:
        return 0  # decrease or static


dataset['trend'] = list(map(increaseOrDecrease, dataset['close'], dataset['future']))

val_split = 0.05
val_split_indices = sorted(dataset.index.values)[-int( val_split * len(sorted(dataset.index.values)) )]
validation_dataset = dataset[(dataset.index >= val_split_indices )]

discard_indices = sorted(dataset.index.values)[-int( 5 )]  # Discard NaN values due to shift
validation_dataset = validation_dataset[(validation_dataset.index <= discard_indices)]

dataset = dataset[(dataset.index < val_split_indices)]

# Drop unnecessary columns
validation_dataset = validation_dataset.drop("low", 1)
validation_dataset = validation_dataset.drop("high", 1)
validation_dataset = validation_dataset.drop("open", 1)
validation_dataset = validation_dataset.drop("future", 1)

dataset = dataset.drop("low", 1)
dataset = dataset.drop("high", 1)
dataset = dataset.drop("open", 1)
dataset = dataset.drop("future", 1)

# =========================================================

# Data Preprocessing
# ---------------------------------------------------------


def preprocess(df):
    for col in df.columns:
        if col != "trend":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
        
    df.dropna(inplace=True)

    # Contains all sequences of required size
    sequences = []
    # Current sequence, deque pushes out old data as new entries are appended
    seq = deque(maxlen=SEQUENCE_LEN)

    for i in df.values:
        seq.append([n for n in i[:-1]])  # Append everything except last column
        if len(seq) == SEQUENCE_LEN:
            sequences.append([np.array(seq), i[-1]])
            # Add the current sequence
    
    # Shuffle order of sequences
    random.shuffle(sequences)

    # Splitting into buys and sells
    buys = []
    sells = []

    for _seq, trend in sequences:
        if trend == 0:
            buys.append([_seq, trend])
        elif trend == 1:
            sells.append([_seq, trend])

    random.shuffle(buys)
    random.shuffle(sells)

    # Equalize the two
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    # Rejoin into main
    sequences = buys + sells
    random.shuffle(sequences)

    # FINALLY Split into features and labels
    X = []
    y = []

    for _seq, trend in sequences:
        X.append(_seq)
        y.append(trend)

    X = np.array(X)

    return X, y


train_X, train_y = preprocess(dataset)
validation_X, validation_y = preprocess(validation_dataset)

# =========================================================

# Creating the Model
# ---------------------------------------------------------

# Sequential Model
model = Sequential()

# Layer 1 :: Sheets = LSTM, Dropout, Normalize
# Input layer from the sequences
model.add(CuDNNLSTM(128, input_shape=(train_X.shape[1:]), return_sequences=True ))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Layer 2 :: Sheets = LSTM, Dropout, Normalize
model.add(CuDNNLSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Layer 3 :: Sheets = LSTM, Dropout, Normalize
# Final layer
model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Layer 4 :: Sheets = Dense, Dropout
# Collection layer, from 128 to 32 nodes
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(2, activation="softmax"))

LEARNING_RATE = 0.001
LEARNING_DECAY = 1e-6

EPOCHS = 10
BATCH_SIZE = 64

opt = Adam(lr=LEARNING_RATE, decay=LEARNING_DECAY)

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train Model
model.fit(
    train_X, train_y,
    batch_size=BATCH_SIZE, epochs=EPOCHS,
    validation_data=(validation_X, validation_y)
)