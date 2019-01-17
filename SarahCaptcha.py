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

symbols = string.ascii_lowercase + "0123456789" # All symbols captcha can contain
num_pics = 1

def preprocess_data():
    X = np.zeros((num_pics, 33, 60))
    y = np.zeros((5, 1, 36))

    # Read image as grayscale
    pic = cv2.imread('0_w5db.png')
    img = cv2.imread(('0_w5db.png'), cv2.IMREAD_GRAYSCALE)
    pic_target = pic[:-4]
 
    # Define targets and code them using OneHotEncoding
    targs = np.zeros((5, 36))
    #for j, l in enumerate(pic_target):
    #    ind = symbols.find(l)
    #    targs[j, ind] = 1
    X[0] = img
    y[:, 0] = targs
    
    # Return final data
    return X, y

X, y = preprocess_data()

