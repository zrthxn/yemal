import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATASET_DIR = "D:\Datasets\PetImages"
CATEGORIES = [ "Dog", "Cat" ]

def test_directory():
    # show one image from each directory to test paths
    for category in CATEGORIES:
        datadir = os.path.join(DATASET_DIR, category)
        for file in os.listdir(datadir):
            try:
                fileread = cv2.imread(os.path.join(datadir, file), cv2.IMREAD_GRAYSCALE)
                plt.imshow(fileread, cmap="gray")
                plt.show()
            except Exception as e:
                print('Error in file:',os.listdir(datadir).index(file), e)
                pass
            break

test_directory()

import random
import pickle

from IPython.display import clear_output

IMG_SIZE_X = 100
IMG_SIZE_Y = 80

training_data = []

def load_training_data():
    # scan each directory to build training dataset
    success = 0
    failed = 0
    for category in CATEGORIES:
        datadir = os.path.join(DATASET_DIR, category)
        label_index = CATEGORIES.index(category)
        for file in os.listdir(datadir):
            try:
                fileread = cv2.imread(os.path.join(datadir, file), cv2.IMREAD_GRAYSCALE)
                fileread = cv2.resize(fileread, (IMG_SIZE_X, IMG_SIZE_Y))
                training_data.append([fileread, label_index])
                success+=1
            except Exception as e:
                clear_output()
                print('Error in file:', category, os.listdir(datadir).index(file), e)
                failed+=1
                pass
    print('Dataset read complete:', success, 'successful', failed, 'failed')
    print(len(training_data), 'images scanned')


def save_dataset(_X, _y):
    X_pickle_out = open('Features.pickle', 'wb')
    pickle.dump(_X, X_pickle_out)
    print('Pickled Features')
    
    y_pickle_out = open('labels.pickle', 'wb')
    pickle.dump(_y, y_pickle_out)
    print('Pickled Labels')
    
    X_pickle_out.close()
    y_pickle_out.close()


F = []
l = []

def create_training_dataset():
    # load_training_data()

    # shuffle the training data randomly
    for i in range(0, random.randint(0, 5)):
        random.shuffle(training_data)

    # create numpy arrays out of the lists
    for features, label in training_data:
        F.append(features)
        l.append(label)

    _F = np.array(F).reshape(-1, IMG_SIZE_X, IMG_SIZE_Y, 1)
    save_dataset(_F, l)

create_training_dataset()
print(F[1])

X = pickle.load(open('Features.pickle', 'rb'))
y = pickle.load(open('labels.pickle', 'rb'))

print('Loaded training dataset from pickle')

print(X[1])

# Normalize pixel data btw 0 to 1
# Either use keras.utils.Normalize, or
# Simple desi way
# normalized_X = []
# for x in X:
#     normalized_line = []
#     for c in x:
#         normalized_line.append(c/255.0)
#     normalized_X.append(normalized_line)
# X = np.array(normalized_X)

X = X/255.0

print(X[0][0][:3], '\n...\n', X[0][0][-3:])

# ============================================================================================
# DATASET PREPARATION DONE ------------------------------------------ DATASET PREPARATION DONE
# ============================================================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D, Conv2D

# Create a Sequential model
model = Sequential()
# Add a INPUT 2D convolutional layer of 64 nodes
model.add( Conv2D(64, (3,3), input_shape = X.shape[1:]) )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Add convolutional layer
model.add( Conv2D(64, (3,3)) )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten 2D to 1D and Add final dense layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

# Output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.fit(X, y, batch_size=16, validation_split=0.1, epochs=10)
model.save("64x3_CNN.model")

model = tf.keras.models.load_model()

def prediction_image(path):
    fileread = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    fileread = cv2.resize(fileread, (IMG_SIZE_X, IMG_SIZE_Y))
    return fileread.reshape(-1, IMG_SIZE_X, IMG_SIZE_Y, 1)

PR_PATH = 'predict\\a.jpg'

pr = model.predict([prediction_image(PR_PATH)])

plt.imshow(cv2.imread(PR_PATH, cv2.IMREAD_COLOR))
plt.show()

print(CATEGORIES[int(pr[0][0])])