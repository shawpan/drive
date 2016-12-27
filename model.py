""" Steering angle prediction model for SDCND Behavariol cloning project
"""
import os
import argparse
import json
import csv
import pickle
import cv2
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

def get_model():
    """ Get the designed NN model
    """
    ch, row, col = 3, 45, 160

    model = Sequential()
    model.add(Lambda(lambda x: x/255. - 0.5,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
    # model.add(Convolution2D(3, 1, 1, border_mode="same"))
    # model.add(ELU())
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    return model

def train_data_generator (x, y, batch_size):
    """ Generates (inputs, outputs) for training in batches
    """
    total = (len(y) // batch_size ) * batch_size
    while True:
        start = np.random.randint(0, total - batch_size)
        end = min(start + batch_size, total)
        x_data = np.empty((0,45,160, 3))
        y_data = np.array([])

        for i in np.arange(start, end):
            image = cv2.imread('./data/' + x[i][0])
            image = image[40:130]
            image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
            flipped_image = cv2.flip(image, 1)
            x_data = np.append(x_data, np.array([image]), axis = 0)
            y_data = np.append(y_data, y[i])
            x_data = np.append(x_data, np.array([flipped_image]), axis = 0)
            y_data = np.append(y_data, -1 * y[i])
        yield x_data, y_data

def train():
    """ Train model using train data and save weights and architecture
    """
    with open('data.p', 'rb') as data_file:
        data = pickle.load(data_file)

    x_train, y_train, x_val, y_val = data['x_train'], data['y_train'], data['x_val'], data['y_val']
    print("Train data : (X, Y) ",  x_train.shape, y_train.shape)
    print("Validation data : (X, Y) ",  x_val.shape, y_val.shape)

    nb_epoch = 1
    batch_size = 16
    model = get_model()
    adam = Adam(lr=1e-6)
    model.compile(optimizer=adam, loss="mse")
    model.load_weights('model_best3.h5')
    model.summary()
    #checkpointer = ModelCheckpoint(filepath="model.h5", verbose=1, save_best_only=True)
    history = model.fit_generator(
        train_data_generator(x_train, y_train, batch_size),
        samples_per_epoch = ((len(y_train) // batch_size ) * batch_size) * 2,
        nb_epoch = nb_epoch,
        verbose = 1,
        validation_data = train_data_generator(x_val, y_val, batch_size),
        nb_val_samples = ((len(y_val) // batch_size ) * batch_size) * 2
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
            speed = float(row['speed'])
            steering_angle = round(float(row['steering']),3)
            x.append(row['left'].strip())
            y.append(steering_angle + 0.25)
            x.append(row['center'].strip())
            y.append(steering_angle)
            x.append(row['right'].strip())
            y.append(steering_angle - 0.25)
        x = np.vstack(x)
        y = np.vstack(y)
    print("Total data : (X, Y) ",  x.shape, y.shape)
    # Split into train and test data
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=3234)

    data = {
        'x_train' : x_train,
        'y_train' : y_train,
        'x_val' : x_val,
        'y_val' : y_val
    }
    # Save as a file
    with open('data.p', "wb") as data_file:
        pickle.dump(data, data_file)


if __name__ == '__main__':
    # Build train, validationa nd test data
    #build_data()
    # Start training
    print("Training started")
    train()
