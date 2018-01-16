from keras.callbacks import TensorBoard
import numpy as np
from model import conv_ae

TRAIN = False
batch_size = 32
epochs = 500
PATIENCE = 20


data = np.load('../../Fold_0/data.npz')
x_train,y_train,x_test,y_test = data['x_train'],data['y_train'],data['x_test'],data['y_test']
print 'Data loaded...'
model = conv_ae((300,300,3))
print 'Model loaded...'
if TRAIN:
    model.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='autoencoder')])
    model.save_weights('ae.h5')
else:
    encoder = conv_e((300,300,3))
    model.load_weights('ae.h5')
    for i in range(7):
        params = model.layers[i].get_weights()
        if params != []:
            encoder.layers[count].set_weights([params[0],params[1]])

    encodings = model.predict(x_test)
    print encodings.shape
    m = 1000000
    n_i = 0
    old_m = m
    for i in range(1,encodings.shape[0]):
         m = min(np.linalg.norm(encodings[0]-encodings[i]))
         if old_m != m:
              n_i = i
              old_m = m
    print m,n_i
