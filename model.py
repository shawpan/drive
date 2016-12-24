""" Steering angle prediction model for SDCND Behavariol cloning project
"""
import os
import argparse
import json
import csv
import pickle
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

def get_model():
    """ Get the designed NN model
    """
    ch, row, col = 3, 160, 320

    model = Sequential()
    model.add(Lambda(lambda x: x/255. - 0.5,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


def data_generator (x, y, batch_size):
    """ Generates (inputs, outputs)
    """
    while True:
        start = np.random.randint(low = 0, high = len(y) - batch_size)
        x_data = np.empty((0,3,160,320))
        y_data = y[start:start + batch_size,:]

        index = start
        for i in np.arange(batch_size):
            image = cv2.imread('./data/' + x[index][0])
            image_h, image_w = image.shape[0], image.shape[1]
            image = image.reshape(3, image_h, image_w)
            x_data = np.append(x_data, np.array([image]), axis = 0)
            index += 1
        yield x_data, y_data

def train():
    """ Train model using train data and save weights and architecture
    """
    with open('data.p', 'rb') as data_file:
        data = pickle.load(data_file)

    x_train, y_train, x_val, y_val = data['x_train'], data['y_train'], data['x_val'], data['y_val']
    model = get_model()
    model.fit_generator(
        data_generator(x_train, y_train, 10),
        samples_per_epoch = 10,
        nb_epoch = 2,
        verbose = 2,
        validation_data = data_generator(x_val, y_val,10),
        nb_val_samples = 10
    )
    # Save weights
    model.save_weights("./model.h5", True)
    # Save model architecture
    with open('./model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

def build_data():
    """ Construct training, validation and test data from log file
    """
    # Get center image path and steering angle
    with open('./data/driving_log.csv') as log_file:
        reader = csv.DictReader(log_file)
        x , y = extra_x = [], []
        for row in reader:
            x.append(row['center'])
            y.append(row['steering'])
        x = np.vstack(x)
        y = np.vstack(y)

    # Shuffle data
    # shuffle(x,y, random_state = 27672)
    print("Total data : (X, Y) ",  x.shape, y.shape)
    # Split into train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3234)
    print("Train data : (X, Y) ",  x_train.shape, y_train.shape)
    print("Test data : (X, Y) ",  x_test.shape, y_test.shape)
    # Split train data into train and validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=937)
    print("Train data : (X, Y) ",  x_train.shape, y_train.shape)
    print("Validation data : (X, Y) ",  x_val.shape, y_val.shape)
    data = {
        'x_train' : x_train,
        'y_train' : y_train,
        'x_val' : x_val,
        'y_val' : y_val,
        'x_test' : x_test,
        'y_test' : y_test
    }
    # Save as a file
    with open('data.p', "wb") as data_file:
        pickle.dump(data, data_file)


if __name__ == '__main__':
    # Build train, validationa nd test data
    if not os.path.isfile('data.p'):
        print("Constructing training, validation and test data from log file")
        build_data()
    else:
        print("Training, validation and test data exist already")
    # Start training
    train()
