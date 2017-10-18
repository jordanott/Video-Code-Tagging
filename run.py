import sys
from load_data import load_data
sys.path.append('Model/')
from model import Inception
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

batch_size = 32
epochs = 15

x_train,y_train,x_test,y_test = load_data()
model = Inception((250,250,3),2)


datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        fill_mode='nearest')
        
datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train,
    	    batch_size=batch_size),
    	    steps_per_epoch=x_train.shape[0] // batch_size,
    	    epochs=epochs,
    	    validation_data=(x_test, y_test))
