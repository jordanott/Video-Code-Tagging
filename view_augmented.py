from load_data import load_data
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import numpy as np
import os
# load data
data = np.load('Fold_0/data.npz')
x_train,y_train,x_test,y_test = data['x_train'],data['y_train'],data['x_test'],data['y_test']

x_train = x_train[:10]
# define data preparation
datagen = ImageDataGenerator(
        #width_shift_range=0.1,
        height_shift_range=0.2,
        #zoom_range=0.2,
        fill_mode='nearest')

# fit parameters from data
datagen.fit(x_train)

x = np.zeros((20,300,300,3))
for i in range(20):
    x[i] = x_train[1]

os.makedirs('Pics')
for X_batch, y_batch in datagen.flow(x, np.zeros(20), batch_size=20, save_to_dir='Pics', save_prefix='aug', save_format='png'):
	for i in range(0, 20):
		pass
	break
