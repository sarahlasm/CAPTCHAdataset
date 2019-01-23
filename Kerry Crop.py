import os
from PIL import Image
import numpy as np

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

dir = os.getcwd()
n = 1
x = []
for filename in os.listdir(dir):
    y = []
    if (filename.endswith('.png')):
        im = Image.open(filename)
        im = im.resize((60,33),Image.ANTIALIAS)
        x = capt_process(im)
        nameindex = filename.find('_') + 1
        y.extend(filename[nameindex:nameindex+4])
        for i in range(len(y)):
            f_name = "/Users/1d_lyx/Desktop/Chapman/Interterm 2019/Cropped/" + str(n) + "_" + y[i] + ".jpg"
            pix = np.array(x[i])
            p = np.count_nonzero(pix) / pix.size
            if 0.2 < p < 0.8:
                x[i].save(f_name)
                n += 1

