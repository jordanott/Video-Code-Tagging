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

def load_data_leave_one_out(directory):
    DATA_DIR = '../Data'
    labels_json = json.load(open('datacleaned.json'))
    x_train,x_test,y_train,y_test = [],[],[],[]
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
                label = np.array([1,0])
                LOAD = True
            # the image has partially visible code
            elif index == 1:
                pass
            # the image has hand written code
            elif index == 2:
                pass
            # the image does not contain code
            elif index == 3:
                label = np.array([0,1])
                LOAD = True
            else:
                pass
        if LOAD:
            complete_location = DATA_DIR+file_location.replace('.png','_resized.png')
            img = Image.open(complete_location)
            if img.mode != 'RGB':
               img = img.convert('RGB')
            img = np.asarray(img)
            if directory in complete_location:
                x_test.append(img)
                y_test.append(label)
            else:
                x_train.append(img)
                y_train.append(label)

    return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)


def load_data(split=.8,seed=0,prefix='',load_new=False):
    if load_new:
        DATA_DIR = prefix+'../Data'
        labels_json = json.load(open(prefix+'datacleaned.json'))
        images = []
        labels = []
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
                    labels.append(np.array([1,0,0,0]))
                    LOAD = True
                # the image has partially visible code
                elif index == 1:
                    labels.append(np.array([0,1,0,0]))
                    LOAD = True
                # the image has hand written code
                elif index == 2:
                    labels.append(np.array([0,0,1,0]))
                    LOAD = True
                # the image does not contain code
                elif index == 3:
                    labels.append(np.array([0,0,0,1]))
                    LOAD = True
            if LOAD:
                complete_location = DATA_DIR+file_location.replace('.png','_resized.png')
                img = Image.open(complete_location)
                if img.mode != 'RGB':
                   img = img.convert('RGB')
                img = np.asarray(img)
                images.append(img)
        random.seed(seed)
        images = np.array(images)
        images,labels = shuffle(images,labels)
        split_index = int(len(images)*split)
        x_train,x_test = images[:split_index],images[split_index:]
        y_train,y_test = labels[:split_index],labels[split_index:]

        count = 0
        for i in range(len(x_train)):
            for j in range(len(x_test)):
                dist = np.linalg.norm(x_train[i]-x_test[j])
                if dist == 0:
                    np.delete(x_test,j,0)
                    np.delete(y_test,j,0)
                    count += 1
        print 'Deleted:',count
        np.savez('data',x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
    else:
        data = np.load(prefix+'data.npz')
        x_train,y_train,x_test,y_test = data['x_train'],data['y_train'],data['x_test'],data['y_test']

    return x_train,y_train,x_test,y_test
