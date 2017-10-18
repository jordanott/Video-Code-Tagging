import os
import random
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def shuffle(x,y):
    c = list(zip(x, y))
    random.shuffle(c)
    a, b = zip(*c)
    return np.array(a),np.array(b)

def load_data(split=.8):
    DATA_DIR = '../Data/'
    labels_file = open(DATA_DIR+'labels.txt')
    images = np.empty((10,200,200,3),dtype='uint8')
    labels = []
    i = 0
    for entry in labels_file:
        file_location,w,x,y,z = entry.split(',')
        labels.append(np.array([int(w),int(x),int(y),int(z)]))
        complete_location = DATA_DIR+file_location.replace('.png','_resized.png')
        img = misc.imread(complete_location)
        images[i] = img
        i += 1

    images,labels = shuffle(images,labels)
    split_index = len(images*split)
    x_train,x_test = images[:split_index],images[split_index:]
    y_train,y_test = labels[:split_index],labels[split_index:]

    return x_train,y_train,x_test,y_test
