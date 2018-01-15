from keras.callbacks import TensorBoard
import numpy as np
from model import conv_ae

batch_size = 32
epochs = 500
PATIENCE = 20


data = np.load('../../Fold_0/data.npz')
x_train,y_train,x_test,y_test = data['x_train'],data['y_train'],data['x_test'],data['y_test']
print 'Data loaded...'
model = conv_ae((300,300,3))
print 'Model loaded...'
model.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='autoencoder')])
