import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization

DATADIR = "C:/Users/User/Desktop/yemal/_data/crypto_data"

dataset = pd.read_csv( os.path.join(DATADIR, "ETH-USD.csv"), names=['time', 'low', 'high', 'open', 'close', 'volume'] )
print(dataset.head())
