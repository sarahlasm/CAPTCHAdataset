#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:11:50 2019

@author: sarahlasman
"""
import numpy as np
import os
import cv2
import string

from keras import layers
from keras.models import Model

symbols = string.ascii_lowercase + "0123456789"
num_pics = 201
shape = (33, 60, 1)

def process_data():
    X = np.zeros((num_pics, 33, 60, 1))
    y = np.zeros((4, num_pics, 36))

    for count, pic in enumerate(os.listdir(path='./Final Train Data')):
        address = pic
        if not (address.endswith('.png')):
            continue
        print(address)
        img = cv2.imread((os.path.join('./Final Train Data', pic)), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (60, 33))
        X[count, :, :, 0] = img
        targs = np.zeros((4, 36))
        for num in range(0, 4):
            ind = symbols.find(address[num+2])
            targs[num, ind] = 1
        y[:, count] = targs
    
    # Return final data
    return X, y

X, y = process_data()
X_train, y_train = X[:90], y[:, :90]
X_test, y_test = X[90:], y[:, 90:]

def create_CNN():
    img = layers.Input(shape=shape) 
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img)
    mp1 = layers.MaxPooling2D(padding='same')(conv1)
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    bn = layers.BatchNormalization()(conv2)
    mp2 = layers.MaxPooling2D(padding='same')(bn)

    flat = layers.Flatten()(mp2)
    output = []
    for i in range(4):
        dens1 = layers.Dense(64, activation='relu')(flat)
        drop = layers.Dropout(0.5)(dens1)
        result = layers.Dense(36, activation='sigmoid')(drop)

        output.append(result)
        
    model = Model(img, output)
    model.compile('rmsprop', loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'])
    return model

cnn = create_CNN()

history = cnn.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3]], batch_size=32, epochs=5, validation_split=0.2)

def predict(address):
    img = cv2.imread((os.path.join('./Final Train Data', address)), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (60, 33))
    img_array = np.zeros((1, 33, 60, 1))
    img_array[0, :, :, 0] = img
    prediction = np.array(cnn.predict(img_array))
    prediction = np.reshape(prediction, (4, 36))
    inds = []
    for char in prediction:
        inds.append(np.argmax(char))
    answer = ''
    for i in inds:
        answer += symbols[i]
    return answer

print(predict('25_ge6m.png'))
