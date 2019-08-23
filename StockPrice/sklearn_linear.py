# The most basic Hello World to ML
# Stock price predictor code

# Scikit

import numpy as np
import pandas as pd
import quandl as Quandl
import math
import datetime

import matplotlib.pyplot as plot
from matplotlib import style

from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
style.use('ggplot')
Quandl.ApiConfig.api_key = "jrPdiJ8AcHLHPofzkNCe"

## GET or IMPORT the data
dataset = Quandl.get('WIKI/GOOGL')

## Sort out valueable data i.e. Decide features
dataset = dataset[['Open', 'High', 'Low', 'Close']]

## Create some new data if required 
dataset['PCT_Change'] = ((dataset['Close'] - dataset['Open']) / dataset['Open']) * 100  # Percent change of stock value over one day
dataset['Volatility'] = ((dataset['High'] - dataset['Low']) / dataset['Low']) * 100    # Percent fluctuation of high and low values

predict_value = 'Close' # The value(s) you want to predict i.e. label(s)

## Fill-in outliers i.e. missing data points or NaN values
dataset.fillna(-99999, inplace=True)
predict_range = int(math.ceil(0.01 * len(dataset))) # This number decides what percent of the length of the dataset is the future

dataset = dataset[:-predict_range]
validate_set = dataset[-predict_range:]

dataset['Label'] = dataset[predict_value].shift(-predict_range)
dataset.dropna(inplace=True)

X = np.array(dataset.drop(['Label'], 1))   # Everything except the labels
y = np.array(dataset['Label'])  # The labels
                 
X_recent = X[-predict_range:]
# X = X[:-predict_range]

## Split data for training, testing and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X = preprocessing.scale(X)  # Scale to desired range. In this case, -1 to 1 (helps with accuracy)

## TRAINING AND TESTING
# model = svm.SVR()
model = LinearRegression()
model.fit(X_train, y_train)

confidence = model.score(X_test, y_test)
print("Confidence =", (int(confidence * 10000))/100, '%')

predictions = model.predict(X_recent)

dataset['Forecast'] = np.nan
last_date = dataset.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in predictions:
    i+=90
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]

dataset['Close'].plot()
dataset['Forecast'].plot()
validate_set['Close'].plot()

print(validate_set.tail())
print(predictions)

plot.legend(loc=4)
plot.xlabel('Date')
plot.ylabel('Price')
plot.show()