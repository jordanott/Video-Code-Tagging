from flask import Flask,request
from keras.preprocessing.image import load_img
import random
import sys
import os
app = Flask(__name__)

sys.path.append('../Models/CNN/')
from model import *

@app.route('/')
def hello_world():
    return 'Hello, World!'

def pull_video(url,quality):
    name = str(random.getrandbits(128))
    path = 'Downloads/' + name + quality + '/'
    Q = quality.replace('_','')
    # set up directory for video and images
    if os.mkdir(str(path)):
        return False
    print 'pytube -e mp4 -p "{path}"/ -f "{name}" -r {Q}p {url}'.format(path=path,name=name,url=url,Q=Q)
    # pull video into directory
    if os.system('pytube -e mp4 -p "{path}"/ -f "{name}" -r {Q}p {url}'.format(path=path,name=name,url=url,Q=Q)):
        return False
    video = '{path}/{name}.mp4'.format(path=path,name=name)
    # split video into images
    if os.system('ffmpeg -i "{video}" -r 1 -f image2 "{path}"/%d.png -nostdin'.format(video=video,path=path)):
        return False
    return path


@app.route('/link/<path:link>',methods=['GET'])
def get_link(link):
    print link
    link += '?'
    for key in request.args.keys():
        link += key + '=' + request.args[key]
    print link
    # generate random number for dir name
    success = False
    qualities = ['_720','_480','_360']
    error = 0
    while not success:
        directory = pull_video(link,qualities[error])
        if directory:
            success = True
            break
        error += 1
    if error == len(qualities):
        return False

    # load model and weights
    model = VGG((300,300,3),2)
    model.load_weights('../Fold_0/code_vs_no_code_strict.h5')
    print 'model loaded'
    # load images from new video
    images = np.empty((1,300,300,3))
    for img in os.listdir(directory):
        if img.endswith('png'):
            image = np.array(load_img(directory+img,target_size=(300,300,3))).reshape(1,300,300,3)
            images = np.append(images,image,axis=0)
    # predict images
    predictions = model.predict(images)
    
    return True

if __name__ == "__main__":
    # connect to ip adress
    app.run(host='0.0.0.0')
