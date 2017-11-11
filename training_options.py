from load_data import load_data,load_data_leave_one_out
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
sys.path.append('Models/CNN/')
from model import Inception,VGG
import os

def code_vs_no_code_strict(x_train,y_train,x_test,y_test):
    X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = [],[],[],[]
    for i in range(len(x_train)):
        if y_train[i] == np.array([1,0,0,0]):
            Y_TRAIN.append([1,0])
        elif y_train[i] == np.array([0,0,0,1]):
            Y_TRAIN.append([0,1])
        X_TRAIN.append(x_train[i])

    for i in range(len(x_test)):
        if y_test[i] == np.array([1,0,0,0]):
            Y_TEST.append([1,0])
        elif y_test[i] == np.array([0,0,0,1]):
            Y_TEST.append([0,1])
        X_TEST.append(x_test[i])

    model = VGG((300,300,3),2)
    return X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,model,'code_vs_no_code_strict.h5'

def code_vs_no_code_partially(x_train,y_train,x_test,y_test):
    X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = [],[],[],[]
    for i in range(len(x_train)):
        if y_train[i] == np.array([1,0,0,0]):
            Y_TRAIN.append([1,0])
        elif y_train[i] == np.array([0,1,0,0]):
            Y_TRAIN.append([1,0])
        elif y_train[i] == np.array([0,0,0,1]):
            Y_TRAIN.append([0,1])
        X_TRAIN.append(x_train[i])

    for i in range(len(x_test)):
        if y_test[i] == np.array([1,0,0,0]):
            Y_TEST.append([1,0])
        elif y_test[i] == np.array([0,1,0,0]):
            Y_TEST.append([1,0])
        elif y_test[i] == np.array([0,0,0,1]):
            Y_TEST.append([0,1])
        X_TEST.append(x_test[i])

    model = VGG((300,300,3),2)

    return X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,model,'code_vs_no_code_partially.h5'

def code_vs_no_code_partially_handwritten(x_train,y_train,x_test,y_test):
    X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = [],[],[],[]
    for i in range(len(x_train)):
        if y_train[i] == np.array([1,0,0,0]):
            Y_TRAIN.append([1,0])
        elif y_train[i] == np.array([0,1,0,0]):
            Y_TRAIN.append([1,0])
        elif y_train[i] == np.array([0,0,1,0]):
            Y_TRAIN.append([1,0])
        elif y_train[i] == np.array([0,0,0,1]):
            Y_TRAIN.append([0,1])
        X_TRAIN.append(x_train[i])

    for i in range(len(x_test)):
        if y_test[i] == np.array([1,0,0,0]):
            Y_TEST.append([1,0])
        elif y_test[i] == np.array([0,1,0,0]):
            Y_TEST.append([1,0])
        elif y_test[i] == np.array([0,0,1,0]):
            Y_TEST.append([1,0])
        elif y_test[i] == np.array([0,0,0,1]):
            Y_TEST.append([0,1])
        X_TEST.append(x_test[i])

    model = VGG((300,300,3),2)

    return X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,model,'code_vs_no_code_partially_handwritten.h5'

def handwritten_vs_else(x_train,y_train,x_test,y_test):
    X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = [],[],[],[]
    for i in range(len(x_train)):
        if y_train[i] == np.array([1,0,0,0]):
            Y_TRAIN.append([0,1])
        elif y_train[i] == np.array([0,1,0,0]):
            Y_TRAIN.append([0,1])
        elif y_train[i] == np.array([0,0,1,0]):
            Y_TRAIN.append([1,0])
        elif y_train[i] == np.array([0,0,0,1]):
            Y_TRAIN.append([0,1])
        X_TRAIN.append(x_train[i])

    for i in range(len(x_test)):
        if y_test[i] == np.array([1,0,0,0]):
            Y_TEST.append([0,1])
        elif y_test[i] == np.array([0,1,0,0]):
            Y_TEST.append([0,1])
        elif y_test[i] == np.array([0,0,1,0]):
            Y_TEST.append([1,0])
        elif y_test[i] == np.array([0,0,0,1]):
            Y_TEST.append([0,1])
        X_TEST.append(x_test[i])

    model = VGG((300,300,3),2)

    return X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,model,'handwritten_vs_else.h5'

def all_four(x_train,y_train,x_test,y_test):
    model = VGG((300,300,3),4)
    return x_train,y_train,x_test,y_test,model,'all_four.h5'
