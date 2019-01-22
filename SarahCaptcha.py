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
