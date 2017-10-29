import os
import json
import random
import numpy as np
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt

def shuffle(x,y):
    c = list(zip(x, y))
    random.shuffle(c)
    a, b = zip(*c)
    return np.array(a),np.array(b)

def load_data(split=.8):
    DATA_DIR = '../Data'
    labels_json = json.load(open(DATA_DIR+'/datacleaned.json'))
    images = []#np.array([],dtype='uint8')
    labels = []
    i = 0
    for i in range(len(labels_json)):
        file_location = str(labels_json[i]['fileName'])
        entry_labels = [labels_json[i]['isCodeCount'],labels_json[i]['isPartiallyCodeCount'],
            labels_json[i]['isHandWrittenCount'],labels_json[i]['isNotCodeCount']]
        index = np.argmax(np.array(entry_labels))
        max_value = np.amax(np.array(entry_labels))
        LOAD = False
        if max_value > 0:
            # the image contains code
            if index == 0:
                labels.append(np.array([1,0]))
                LOAD = True
            # the image has partially visible code
            elif index == 1:
                pass
            # the image has hand written code
            elif index == 2:
                pass
            # the image does not contain code
            elif index == 3:
                labels.append(np.array([0,1]))
                LOAD = True
            else:
                pass
        if LOAD:
            complete_location = DATA_DIR+file_location.replace('.png','_resized.png')
            #img = misc.imread(complete_location)
            img = Image.open(complete_location)
            if img.mode != 'RGB':
               img = img.convert('RGB')
            img = np.asarray(img)
            images.append(img)#np.concatenate((images,img),axis=0)
    images = np.array(images)
    images,labels = shuffle(images,labels)
    split_index = int(len(images)*split)
    x_train,x_test = images[:split_index],images[split_index:]
    y_train,y_test = labels[:split_index],labels[split_index:]

    return x_train,y_train,x_test,y_test
