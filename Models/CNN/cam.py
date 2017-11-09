import json
import sys
import numpy as np
from model import VGG
sys.path.append('../../')
from load_data import load_data
import matplotlib.pyplot as plt
from vis.visualization import overlay
from vis.visualization.saliency import visualize_cam

images,labels = load_data(single=True,prefix='../../')
# load model & weights
model = VGG((300,300,3),2)
model.load_weights('../../vgg_weights.h5')
predicitions = model.predict(images)
# accuracy
print np.sum(np.argmax(predicitions,axis=1)==np.argmax(labels,axis=1))/1000.0
for i in range(len(predicitions)):
    code = np.argmax(predicitions[i])
    if code:
        print 'Not code'
    else:
        print 'Code'
    cam = visualize_cam(model,len(model.layers)-1,code,images[i].reshape(1,300,300,3))

    img = images[i]

    plt.imshow(overlay(cam,img))
    plt.show()
