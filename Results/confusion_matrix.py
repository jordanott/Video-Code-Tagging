import sys
import numpy as np
sys.path.append('../Models/CNN/')
from model import Inception,VGG
sys.path.append('../')
from training_options import *
from load_data import load_custom


weights = ['code_vs_no_code_strict.h5','code_vs_no_code_partially.h5','code_vs_no_code_partially_handwritten.h5','handwritten_vs_else.h5','all_four.h5']

# class options
two_options = {0:'code',1:'nc'}
four_options = {0:'code',1:'partially',2:'handwritten',3:'nc'}
# training option functions
functions = [code_vs_no_code_strict,code_vs_no_code_partially,code_vs_no_code_partially_handwritten,handwritten_vs_else,all_four]

for f in functions:
    for fold in range(0,5):
        fold_dir = 'Fold_'+str(fold)+'/'
        # load data from file
        X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = load_data(prefix='../'+fold_dir)
        # get data, model and weights file name from training options function
        x_train,y_train,x_test,y_test,model,weights = f(X_TRAIN,Y_TRAIN,X_TEST,Y_TEST)
        # load weights file
        model.load_weights('../' + fold_dir + weights)
        # predict classes for testing images
        predicitions = model.predict(x_test)

        print np.sum(np.argmax(predicitions,axis=1) == np.argmax(y_test,axis=1))
        print predicitions.shape[0]
        print np.sum(np.argmax(predicitions,axis=1) == np.argmax(y_test,axis=1))/float(predicitions.shape[0])
