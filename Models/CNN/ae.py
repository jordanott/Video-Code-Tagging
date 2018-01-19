from keras.preprocessing.image import load_img
from keras.callbacks import TensorBoard
import numpy as np
from model import conv_ae,conv_e
import os

TRAIN = True
batch_size = 32
epochs = 500
PATIENCE = 20

images = np.empty((1,300,300,3))
for subdir,dirs,files in os.walk('../../../Data/'):
    for img in files:
        if img.endswith('_resized.png'):
            img_path = os.path.join(subdir,img)
            image = np.array(load_img(img_path,target_size=(300,300,3))).reshape(1,300,300,3)
            images = np.append(images,image,axis=0)
print 'Data loaded...', images.shape
model = conv_ae((300,300,3))
print 'Model loaded...'
if TRAIN:
    import time
    start = time.time()
    model.fit(images, images,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(images, images),
                callbacks=[TensorBoard(log_dir='autoencoder')])
    print time.time() - start
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
