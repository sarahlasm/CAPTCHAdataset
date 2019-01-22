import os
import numpy as np
from PIL import Image
import string
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization
import cv2

dir = os.getcwd()
'''
for filename in os.listdir(dir):
    if (filename == '4_c.jpg'):
        im = np.asarray(Image.open(filename))
        index = filename.find("_") + 1
        y.append(filename[index])
'''
num_pics = 40005
symbols = string.ascii_lowercase + "0123456789"

def process_data():
    X = np.zeros((num_pics, 15, 30, 1))
    y = np.zeros((1, num_pics, 36))

    for count, pic in enumerate(dir):
        address = pic
        if not (address.endswith('.jpg')):
            continue
        print(address)
        img = cv2.imread((os.path.join(dir, pic)), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (30, 15))
        X[count, :, :, 0] = img
        targs = np.zeros((1, 36))
        index = symbols.find(address.find("_") + 1)
        targs[0,index] = 1
        y[:, count] = targs
    
    # Return final data
    return X, y
    
def CNN():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(15,30,1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(36, activation='softmax'))
    model.add(Dropout(0.5))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

cnn = CNN()

X,y = process_data()
X_train, y_train = X[:36000],y[:,:36000]
X_test, y_test = X[36000:],y[:,36000:]

history = cnn.fit(X_train, [y_train[0]], batch_size = 64, validation_data=(X_test, y_test[0]), epochs=5)

