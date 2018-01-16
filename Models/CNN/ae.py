from keras.callbacks import TensorBoard
import numpy as np
from model import conv_ae,conv_e

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
            encoder.layers[i].set_weights([params[0],params[1]])

    encodings = encoder.predict(x_test)
    print encodings.shape

    def plot(img1,img2):
        f,ax = plt.subplots(1,2)
        ax[0].imshow(img1)
        ax[1].imshow(img2)
        plt.show()
    
    def dist(vector):
        return np.linalg.norm(encodings[0] - vector[0])

    items = []
    for i,j in enumerate(encodings):
        items.append([j,i])
 
    items = sorted(items,key=dist)
    print items[0][1]
    print items[1][1]
    print items[len(items)-1][1]    
