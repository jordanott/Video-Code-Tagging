from load_data import load_data,load_data_leave_one_out
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from training_options import *
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import json
import sys
sys.path.append('Models/CNN/')
from model import Inception,VGG
import os

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--load',action="store_true")
parser.add_argument('--show',action="store_true")
parser.add_argument('--train',action="store_true")
parser.add_argument('--leave_one_out',action="store_true")
args = parser.parse_args()
# visualize the images and choices
LOAD = args.load
TRAIN = args.train
VISUALIZE = args.show
LEAVE_ONE_OUT = args.leave_one_out

batch_size = 32
epochs = 500
PATIENCE = 20

if TRAIN:
	if LEAVE_ONE_OUT:
		os.mkdir('Weights')
		os.mkdir('TensorBoard')
		log = open('log.txt','w')
		videos = next(os.walk('../Data/'))[1]
		count = 0
		for directory in videos:
			count += 1
			x_train,y_train,x_test,y_test = load_data_leave_one_out(directory)
			log.write(str(count)+','+directory+',Train C/NC:'+str(np.sum(y_train[:,0]))+'/'+str(np.sum(y_train[:,1]))+
				',Test C/NC:'+str(np.sum(y_test[:,0]))+'/'+str(np.sum(y_test[:,1])))
			# load pretrained network
			model = Inception((300,300,3),2)
			# data augmentation
			datagen = ImageDataGenerator(
				width_shift_range=0.1,
				height_shift_range=0.1,
				zoom_range=0.2,
				fill_mode='nearest')
			datagen.fit(x_train)
			# Callbacks
			tb = TensorBoard(log_dir='TensorBoard/{}'.format(count),histogram_freq=0, write_graph=True, write_images=True)
			es = EarlyStopping(monitor='val_acc', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')
			w = ModelCheckpoint("Weights/{}.h5".format(count),monitor='val_acc', verbose=1, save_best_only=True, mode='max')
			# Training
			history = model.fit_generator(datagen.flow(x_train, y_train,
				batch_size=batch_size),
				steps_per_epoch=x_train.shape[0] // batch_size,
				epochs=epochs,
				validation_data=(x_test, y_test),
				callbacks=[tb,es,w])

			log.write(',ValAcc:'+"{0:.2f}".format(100*max(history.history['val_acc']))+'\n')

	else:
		all_folds_acc = {
		'code_vs_no_code_strict.h5':[],
		'code_vs_no_code_partially.h5':[],
		'code_vs_no_code_partially_handwritten.h5':[],
		'handwritten_vs_else.h5':[],
		'all_four.h5':[]
		}
		all_folds_ds = {
		'code_vs_no_code_strict.h5':[[0,0],[0,0]],
		'code_vs_no_code_partially.h5':[[0,0],[0,0]],
		'code_vs_no_code_partially_handwritten.h5':[[0,0],[0,0]],
		'handwritten_vs_else.h5':[[0,0],[0,0]],
		'all_four.h5':[[0,0,0,0],[0,0,0,0]]
		}
		for fold in range(5):
			os.mkdir('Fold_'+str(fold))
			# load data
			X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = load_data(fold=fold,load_new=True)
			print 'Code samples:',np.sum(Y_TRAIN[:,0]),'No C samples:',np.sum(Y_TRAIN[:,1])
			print 'Train:',len(X_TRAIN),'Test:',len(X_TEST)

			functions = [code_vs_no_code_strict,code_vs_no_code_partially,code_vs_no_code_partially_handwritten,handwritten_vs_else,all_four]
			for f in functions:
				x_train,y_train,x_test,y_test,model,weights = f(X_TRAIN,Y_TRAIN,X_TEST,Y_TEST)

				train_break_down = ', Train C/P/H/NC:'
				for i in range(len(y_train[0])):
					train_break_down += str(np.sum(y_train[:,i])) +'/'
				test_break_down = ', Test C/P/H/NC:'
				for i in range(len(y_test[0])):
					test_break_down += str(np.sum(y_test[:,i])) +'/'

				for i in range(y_train.shape[1]):
					all_folds_ds[weights][0][i] += np.sum(y_train[:,i])
				for i in range(y_test.shape[1]):
					all_folds_ds[weights][1][i] += np.sum(y_test[:,i])

				log = open('Fold_'+str(fold)+'/log.txt','a')
				log.write(weights+train_break_down[:-1]+test_break_down[:-1])
				log.close()
				datagen = ImageDataGenerator(
					width_shift_range=0.1,
					height_shift_range=0.1,
					zoom_range=0.2,
					fill_mode='nearest')
				datagen.fit(x_train)

				# Callbacks
				tb = TensorBoard(log_dir='TensorBoard/'+'Fold_'+str(fold)+'/'+weights.replace('.h5',''),histogram_freq=0, write_graph=True, write_images=True)
				es = EarlyStopping(monitor='val_acc', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')
				w = ModelCheckpoint('Fold_'+str(fold)+'/'+weights,monitor='val_acc', verbose=1, save_best_only=True, mode='max')
				# Training
				history = model.fit_generator(datagen.flow(x_train, y_train,
					batch_size=batch_size),
					steps_per_epoch=x_train.shape[0] // batch_size,
					epochs=epochs,
					validation_data=(x_test, y_test),
					callbacks=[tb,es,w])

				log = open('Fold_'+str(fold)+'/log.txt','a')
				log.write(', ValAcc:'+"{0:.2f}".format(100*max(history.history['val_acc']))+'\n')
				log.close()
				all_folds[weights].append(100*max(history.history['val_acc']))
		with open('all_folds_acc.json', 'w') as fp:
			json.dump(all_folds_acc, fp)

		with open('all_folds_ds.json', 'w') as fp:
			json.dump(all_folds_ds, fp)

		with open('latex.txt', 'w') as latex:
			for key in all_folds_ds.keys():
				line = key.replace('.h5','')
				line += ' & '
				# train
				for i in range(len(all_folds_ds[key][0])):
					line += all_folds_ds[key][0] + ','
				line = line[:-1] + ' & '
				# test
				for i in range(len(all_folds_ds[key][1])):
					line += all_folds_ds[key][1] + ','
				line = line[:-1] + ' \\\\\n'
elif LOAD:
	# load data
	x_train,y_train,x_test,y_test = load_data()
	print 'Code samples:',np.sum(y_train[:,0])
	print 'No C samples:',np.sum(y_train[:,1])
	model = Inception((300,300,3),2)

	model.load_weights('code_tagger_weights.h5')
	print 'Code samples:',np.sum(y_test[:,0])
	print 'No C samples:',np.sum(y_test[:,1])
	correct = 0
	for i in range(len(x_test)):
		prediction = np.argmax(model.predict(x_test[i].reshape(1,300,300,3)))
		actual = np.argmax(y_test[i])
		if prediction == actual:
			correct += 1
		if (i+1) % 300 == 0:
			print correct/float(i)
		if VISUALIZE:
			print prediction
			print actual
			plt.imshow(x_test[i])
			plt.show()

	print 'Accuracy:',correct/float(len(x_test))
