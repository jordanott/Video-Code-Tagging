import sys
from load_data import load_data
sys.path.append('Models/CNN/')
from model import Inception
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
import numpy as np

batch_size = 32
epochs = 5
PATIENCE = 2

# load data
x_train,y_train,x_test,y_test = load_data()
print 'Code samples:',np.sum(y_train[:,0])
print 'No C samples:',np.sum(y_train[:,1])
model = Inception((300,300,3),2)

datagen = ImageDataGenerator(
	width_shift_range=0.1,
	height_shift_range=0.1,
	zoom_range=0.2,
	fill_mode='nearest')
datagen.fit(x_train)

# Callbacks
tb = TensorBoard(log_dir='TensorBoard',histogram_freq=0, write_graph=True, write_images=True)
es = EarlyStopping(monitor='val_acc', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')
w = ModelCheckpoint("code_tagger_weights.h5",monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# Training 
model.fit_generator(datagen.flow(x_train, y_train,
	batch_size=batch_size),
	steps_per_epoch=x_train.shape[0] // batch_size,
	epochs=epochs,
	validation_data=(x_test, y_test),
	callbacks=[tb,es,w])
