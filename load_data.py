import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def load_data():
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
    return images, labels

i,j = load_data()

plt.imshow(i[1])
plt.show()
