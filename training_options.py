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

def java_python(x_train,y_train,x_test,y_test,px_train,py_train,px_test,py_test):
    X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = [],[],[],[]
    # java train
    for i in range(len(x_train)):
        if np.all(y_train[i] == np.array([1,0,0,0])):
            Y_TRAIN.append([1,0])
            X_TRAIN.append(x_train[i])
    # java test
    for i in range(len(x_test)):
        if np.all(y_test[i] == np.array([1,0,0,0])):
            Y_TEST.append([1,0])
            X_TEST.append(x_test[i])
    # python train
    for i in range(len(px_train)):
        if np.all(py_train[i] == np.array([1,0,0,0])):
            Y_TRAIN.append([0,1])
            X_TRAIN.append(px_train[i])
    # python test
    for i in range(len(px_test)):
        if np.all(py_test[i] == np.array([1,0,0,0])):
            Y_TEST.append([0,1])
            X_TEST.append(px_test[i])

    model = VGG((300,300,3),2)
    return np.array(X_TRAIN),np.array(Y_TRAIN),np.array(X_TEST),np.array(Y_TEST),model,'java_python.h5'

def java_python_no_code(x_train,y_train,x_test,y_test,px_train,py_train,px_test,py_test):
    X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = [],[],[],[]
    # java train
    for i in range(len(x_train)):
        if np.all(y_train[i] == np.array([1,0,0,0])):
            Y_TRAIN.append([1,0,0])
            X_TRAIN.append(x_train[i])
        elif np.all(y_train[i] == np.array([0,0,0,1])):
            Y_TRAIN.append([0,0,1])
            X_TRAIN.append(x_train[i])
    # java test
    for i in range(len(x_test)):
        if np.all(y_test[i] == np.array([1,0,0,0])):
            Y_TEST.append([1,0,0])
            X_TEST.append(x_test[i])
        elif np.all(y_test[i] == np.array([0,0,0,1])):
            Y_TEST.append([0,0,1])
            X_TEST.append(x_test[i])

    # python train
    for i in range(len(px_train)):
        if np.all(py_train[i] == np.array([1,0,0,0])):
            Y_TRAIN.append([0,1,0])
            X_TRAIN.append(px_train[i])
        elif np.all(py_train[i] == np.array([0,0,0,1])):
            Y_TRAIN.append([0,0,1])
            X_TRAIN.append(px_train[i])
    # python test
    for i in range(len(px_test)):
        if np.all(py_test[i] == np.array([1,0,0,0])):
            Y_TEST.append([0,1,0])
            X_TEST.append(px_test[i])
        elif np.all(py_test[i] == np.array([0,0,0,1])):
            Y_TEST.append([0,0,1])
            X_TEST.append(px_test[i])

    model = VGG((300,300,3),3)
    return np.array(X_TRAIN),np.array(Y_TRAIN),np.array(X_TEST),np.array(Y_TEST),model,'java_python_no_code.h5'

def java_python_pv_no_code(x_train,y_train,x_test,y_test,px_train,py_train,px_test,py_test):
    X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = [],[],[],[]
    # java train
    for i in range(len(x_train)):
        if np.all(y_train[i] == np.array([1,0,0,0])):
            Y_TRAIN.append([1,0,0])
            X_TRAIN.append(x_train[i])
        elif np.all(y_train[i] == np.array([0,1,0,0])):
            Y_TRAIN.append([1,0,0])
            X_TRAIN.append(x_train[i])
        elif np.all(y_train[i] == np.array([0,0,0,1])):
            Y_TRAIN.append([0,0,1])
            X_TRAIN.append(x_train[i])
    # java test
    for i in range(len(x_test)):
        if np.all(y_test[i] == np.array([1,0,0,0])):
            Y_TEST.append([1,0,0])
            X_TEST.append(x_test[i])
        elif np.all(y_test[i] == np.array([0,1,0,0])):
            Y_TEST.append([1,0,0])
            X_TEST.append(x_test[i])
        elif np.all(y_test[i] == np.array([0,0,0,1])):
            Y_TEST.append([0,0,1])
            X_TEST.append(x_test[i])

    # python train
    for i in range(len(px_train)):
        if np.all(py_train[i] == np.array([1,0,0,0])):
            Y_TRAIN.append([0,1,0])
            X_TRAIN.append(px_train[i])
        elif np.all(py_train[i] == np.array([0,1,0,0])):
            Y_TRAIN.append([0,1,0])
            X_TRAIN.append(px_train[i])
        elif np.all(py_train[i] == np.array([0,0,0,1])):
            Y_TRAIN.append([0,0,1])
            X_TRAIN.append(px_train[i])
    # python test
    for i in range(len(px_test)):
        if np.all(py_test[i] == np.array([1,0,0,0])):
            Y_TEST.append([0,1,0])
            X_TEST.append(px_test[i])
        elif np.all(py_test[i] == np.array([0,1,0,0])):
            Y_TEST.append([0,1,0])
            X_TEST.append(px_test[i])
        elif np.all(py_test[i] == np.array([0,0,0,1])):
            Y_TEST.append([0,0,1])
            X_TEST.append(px_test[i])

    model = VGG((300,300,3),3)
    return np.array(X_TRAIN),np.array(Y_TRAIN),np.array(X_TEST),np.array(Y_TEST),model,'java_python_pv_no_code.h5'

def code_vs_no_code_strict(x_train,y_train,x_test,y_test):
    X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = [],[],[],[]
    for i in range(len(x_train)):
        if np.all(y_train[i] == np.array([1,0,0,0])):
            Y_TRAIN.append([1,0])
            X_TRAIN.append(x_train[i])
        elif np.all(y_train[i] == np.array([0,0,0,1])):
            Y_TRAIN.append([0,1])
            X_TRAIN.append(x_train[i])

    for i in range(len(x_test)):
        if np.all(y_test[i] == np.array([1,0,0,0])):
            Y_TEST.append([1,0])
            X_TEST.append(x_test[i])
        elif np.all(y_test[i] == np.array([0,0,0,1])):
            Y_TEST.append([0,1])
            X_TEST.append(x_test[i])

    model = VGG((300,300,3),2)
    return np.array(X_TRAIN),np.array(Y_TRAIN),np.array(X_TEST),np.array(Y_TEST),model,'code_vs_no_code_strict.h5'

def code_vs_no_code_partially(x_train,y_train,x_test,y_test):
    X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = [],[],[],[]
    for i in range(len(x_train)):
        if np.all(y_train[i] == np.array([1,0,0,0])):
            Y_TRAIN.append([1,0])
            X_TRAIN.append(x_train[i])
        elif np.all(y_train[i] == np.array([0,1,0,0])):
            Y_TRAIN.append([1,0])
            X_TRAIN.append(x_train[i])
        elif np.all(y_train[i] == np.array([0,0,0,1])):
            Y_TRAIN.append([0,1])
            X_TRAIN.append(x_train[i])

    for i in range(len(x_test)):
        if np.all(y_test[i] == np.array([1,0,0,0])):
            Y_TEST.append([1,0])
            X_TEST.append(x_test[i])
        elif np.all(y_test[i] == np.array([0,1,0,0])):
            Y_TEST.append([1,0])
            X_TEST.append(x_test[i])
        elif np.all(y_test[i] == np.array([0,0,0,1])):
            Y_TEST.append([0,1])
            X_TEST.append(x_test[i])

    model = VGG((300,300,3),2)

    return np.array(X_TRAIN),np.array(Y_TRAIN),np.array(X_TEST),np.array(Y_TEST),model,'code_vs_no_code_partially.h5'

def code_vs_no_code_partially_handwritten(x_train,y_train,x_test,y_test):
    X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = [],[],[],[]
    for i in range(len(x_train)):
        if np.all(y_train[i] == np.array([1,0,0,0])):
            Y_TRAIN.append([1,0])
        elif np.all(y_train[i] == np.array([0,1,0,0])):
            Y_TRAIN.append([1,0])
        elif np.all(y_train[i] == np.array([0,0,1,0])):
            Y_TRAIN.append([1,0])
        elif np.all(y_train[i] == np.array([0,0,0,1])):
            Y_TRAIN.append([0,1])
        X_TRAIN.append(x_train[i])

    for i in range(len(x_test)):
        if np.all(y_test[i] == np.array([1,0,0,0])):
            Y_TEST.append([1,0])
        elif np.all(y_test[i] == np.array([0,1,0,0])):
            Y_TEST.append([1,0])
        elif np.all(y_test[i] == np.array([0,0,1,0])):
            Y_TEST.append([1,0])
        elif np.all(y_test[i] == np.array([0,0,0,1])):
            Y_TEST.append([0,1])
        X_TEST.append(x_test[i])

    model = VGG((300,300,3),2)

    return np.array(X_TRAIN),np.array(Y_TRAIN),np.array(X_TEST),np.array(Y_TEST),model,'code_vs_no_code_partially_handwritten.h5'

def handwritten_vs_else(x_train,y_train,x_test,y_test):
    X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = [],[],[],[]
    for i in range(len(x_train)):
        if np.all(y_train[i] == np.array([1,0,0,0])):
            Y_TRAIN.append([0,1])
        elif np.all(y_train[i] == np.array([0,1,0,0])):
            Y_TRAIN.append([0,1])
        elif np.all(y_train[i] == np.array([0,0,1,0])):
            Y_TRAIN.append([1,0])
        elif np.all(y_train[i] == np.array([0,0,0,1])):
            Y_TRAIN.append([0,1])
        X_TRAIN.append(x_train[i])

    for i in range(len(x_test)):
        if np.all(y_test[i] == np.array([1,0,0,0])):
            Y_TEST.append([0,1])
        elif np.all(y_test[i] == np.array([0,1,0,0])):
            Y_TEST.append([0,1])
        elif np.all(y_test[i] == np.array([0,0,1,0])):
            Y_TEST.append([1,0])
        elif np.all(y_test[i] == np.array([0,0,0,1])):
            Y_TEST.append([0,1])
        X_TEST.append(x_test[i])

    model = VGG((300,300,3),2)

    return np.array(X_TRAIN),np.array(Y_TRAIN),np.array(X_TEST),np.array(Y_TEST),model,'handwritten_vs_else.h5'

def handwritten_vs_no_code(x_train,y_train,x_test,y_test):
    X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = [],[],[],[]
    for i in range(len(x_train)):
        if np.all(y_train[i] == np.array([0,0,1,0])):
            Y_TRAIN.append([1,0])
            X_TRAIN.append(x_train[i])
        elif np.all(y_train[i] == np.array([0,0,0,1])):
            Y_TRAIN.append([0,1])
            X_TRAIN.append(x_train[i])

    for i in range(len(x_test)):
        if np.all(y_test[i] == np.array([0,0,1,0])):
            Y_TEST.append([1,0])
            X_TEST.append(x_test[i])
        elif np.all(y_test[i] == np.array([0,0,0,1])):
            Y_TEST.append([0,1])
            X_TEST.append(x_test[i])

    model = VGG((300,300,3),2)

    return np.array(X_TRAIN),np.array(Y_TRAIN),np.array(X_TEST),np.array(Y_TEST),model,'handwritten_vs_no_code.h5'

def all_four(x_train,y_train,x_test,y_test):
    model = VGG((300,300,3),4)
    return x_train,y_train,x_test,y_test,model,'all_four.h5'
