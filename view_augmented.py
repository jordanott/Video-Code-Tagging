from load_data import load_data
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import numpy as np
import os
# load data
X_train, y_train,X_test,y_test = load_data()
# define data preparation
datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        fill_mode='nearest')

# fit parameters from data
datagen.fit(X_train)

x = np.zeros((10,200,200,3))
for i in range(10):
    x[i] = X_train[1]

os.makedirs('Pics')
for X_batch, y_batch in datagen.flow(x, np.zeros(10), batch_size=10, save_to_dir='Pics', save_prefix='aug', save_format='png'):
	for i in range(0, 20):
		pass
	break
