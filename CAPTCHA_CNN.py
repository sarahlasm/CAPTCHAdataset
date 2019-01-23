import os
import numpy as np
from PIL import Image
import string
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization
import cv2

num_pics = 20262
symbols = string.ascii_lowercase + "0123456789"

def process_data():
    X = np.zeros((num_pics, 15, 30, 1))
    y = np.zeros((1, num_pics, 36))

    for count, pic in enumerate(os.listdir(path='./Cropped')):
        address = pic
        if count >= num_pics:
            break
        if not (address.endswith('.jpg')):
            continue
        img = cv2.imread((os.path.join('./Cropped', pic)), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (30, 15))
        X[count, :, :, 0] = img
        targs = np.zeros((1, 36))
        index = symbols.find(address[address.find("_") + 1])
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
X_train, y_train = X[:18000],y[:,:18000]
X_test, y_test = X[18000:],y[:,18000:]

history = cnn.fit(X_train, [y_train[0]], batch_size = 64, validation_data=(X_test, y_test[0]), epochs=5)

THRESHOLD = 165
LUT = [0]*THRESHOLD + [1]*(256 - THRESHOLD)

def capt_process(capt):
    capt_gray = capt.convert("L")
    capt_bw = capt_gray.point(LUT, "1")

    capt_per_char_list = []
    for i in range(4):
        x = 2 + i * 14
        y = 2
        capt_per_char = capt_bw.crop((x, y, x + 15, y + 30))
        capt_per_char_list.append(capt_per_char)

    return capt_per_char_list

def predict4(address):
    im = Image.open(os.path.join('./Final Train Data', address))
    im = im.resize((60,33),Image.ANTIALIAS)
    im_list = []
    im_list = capt_process(im)
    data = []
    for i in range(0, 4):
        im_list[i].save("./Cropped/predict.jpg")
        data.append(predict("predict.jpg"))
    answer = ''
    for i in data:
        answer += i
    return answer

def predict(address):
    img = cv2.imread((os.path.join('./Cropped', address)), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (30, 15))
    img_array = np.zeros((1, 15, 30, 1))
    img_array[0, :, :, 0] = img
    prediction = np.array(cnn.predict(img_array))
    prediction = np.reshape(prediction, (1, 36))
    inds = []
    for char in prediction:
        inds.append(np.argmax(char))
    answer = ''
    for i in inds:
        answer += symbols[i]
    return answer

print(predict4('2_nwp8.png'))
