from load_data import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from training_options import *
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import sys
sys.path.append('Models/CNN/')
from model import Inception,VGG
import os

random.seed(0)

batch_size = 32
epochs = 500
PATIENCE = 30

all_folds_acc = {
'java_python.h5':[],
'java_python_no_code.h5':[],
'java_python_pv_no_code.h5':[]
}
all_folds_ds = {
'java_python.h5':[[0,0],[0,0]],
'java_python_no_code.h5':[[0,0,0],[0,0,0]],
'java_python_pv_no_code.h5':[[0,0,0],[0,0,0]]
}
for fold in range(5):
	#os.mkdir('jp_Fold_'+str(fold))
	# load data
	X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = load_from_npz('jp_Fold_{k}/Java.npz'.format(k=fold))
        X_TRAIN,X_TEST = np.squeeze(X_TRAIN),np.squeeze(X_TEST)
        pX_TRAIN,pY_TRAIN,pX_TEST,pY_TEST = load_from_npz('jp_Fold_{k}/Python.npz'.format(k=fold))
        pX_TRAIN,pX_TEST = np.squeeze(pX_TRAIN),np.squeeze(pX_TEST)

	print 'Java train:',X_TRAIN.shape
        print 'Python train:',pX_TRAIN.shape

	functions = [java_python,java_python_no_code,java_python_pv_no_code]
	for f in functions:
		x_train,y_train,x_test,y_test,model,weights = f(X_TRAIN,Y_TRAIN,X_TEST,Y_TEST,pX_TRAIN,pY_TRAIN,pX_TEST,pY_TEST)

		train_break_down = ', Train J/P/NC:'
		for i in range(len(y_train[0])):
			train_break_down += str(np.sum(y_train[:,i])) +'/'
		test_break_down = ', Test J/P/NC:'
		for i in range(len(y_test[0])):
			test_break_down += str(np.sum(y_test[:,i])) +'/'

		for i in range(y_train.shape[1]):
			all_folds_ds[weights][0][i] += np.sum(y_train[:,i])
		for i in range(y_test.shape[1]):
			all_folds_ds[weights][1][i] += np.sum(y_test[:,i])

		log = open('jp_Fold_'+str(fold)+'/log.txt','a')
		log.write(weights+train_break_down[:-1]+test_break_down[:-1])
		log.close()
		datagen = ImageDataGenerator(
			width_shift_range=0.1,
			height_shift_range=0.1,
			zoom_range=0.2,
			fill_mode='nearest')
		datagen.fit(x_train)

		# Callbacks
		tb = TensorBoard(log_dir='TensorBoard/'+'jp_Fold_'+str(fold)+'/'+weights.replace('.h5',''),histogram_freq=0, write_graph=True, write_images=True)
		es = EarlyStopping(monitor='val_acc', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')
		w = ModelCheckpoint('jp_Fold_'+str(fold)+'/'+weights,monitor='val_acc', verbose=1, save_best_only=True, mode='max')
		# Training
		history = model.fit_generator(datagen.flow(x_train, y_train,
			batch_size=batch_size),
			steps_per_epoch=x_train.shape[0] // batch_size,
			epochs=epochs,
			validation_data=(x_test, y_test),
			callbacks=[tb,es,w])

		log = open('jp_Fold_'+str(fold)+'/log.txt','a')
		log.write(', ValAcc:'+"{0:.2f}".format(100*max(history.history['val_acc']))+'\n')
		log.close()
		all_folds_acc[weights].append(100*max(history.history['val_acc']))

with open('jp_all_folds_acc.json', 'w') as fp:
	json.dump(all_folds_acc, fp)

with open('jp_all_folds_ds.json', 'w') as fp:
	json.dump(all_folds_ds, fp)

with open('jp_latex.txt', 'w') as latex:
	for key in all_folds_ds.keys():
		line = key.replace('.h5','')
		line += ' & '
		# train
		for i in range(len(all_folds_ds[key][0])):
			line += str(all_folds_ds[key][0][i]) + ','
		line = line[:-1] + ' & '
		# test
		for i in range(len(all_folds_ds[key][1])):
			line += str(all_folds_ds[key][1][i]) + ','
		line = line[:-1] + ' \\\\\n'
		latex.write(line)
