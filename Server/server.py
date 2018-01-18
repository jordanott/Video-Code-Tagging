from flask import Flask,request,jsonify
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

def make_timestamps(times_list):
    times = []
    start_index = 0
    index = 0
    while index < len(times_list):
            while index+1 != len(times_list) and times_list[index+1] == (times_list[index]+1):
                index += 1
            times.append({"startTime": times_list[start_index],"endTime": times_list[index]})
            index += 1
            start_index = index
    return times

@app.route('/link/<path:link>',methods=['GET'])
def get_link(link):
    import time
    start = time.time()
    json_obj = {
    'youtubeId':request.args['v'],
    'times':[]
    }

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
    print time.time() - start
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
    print images.shape
    # predict images
    predictions = model.predict(images)

    print np.sum(np.argmax(predictions,axis=1))
    code_times = np.squeeze(np.array(np.where(np.argmax(predictions,axis=1)==0)))
    print code_times.shape
    code_images = images[code_times]
    print code_images.shape
    json_obj['times'] = make_timestamps(code_times)
    print json_obj
    # autoencoder
    encoder = conv_e((300,300,3))
    autoencoder = conv_ae((300,300,3))
    autoencoder.load_weights('../Models/CNN/ae.h5')
    for i in range(7):
        params = autoencoder.layers[i].get_weights()
        if params != []:
            encoder.layers[i].set_weights([params[0],params[1]])

    encodings = encoder.predict(code_images)
    print encodings.shape
    print time.time() - start
    def plot(img1,img2):
        f,ax = plt.subplots(1,2)
        ax[0].imshow(img1)
        ax[1].imshow(img2)
        plt.show()

    def dist(vector):
        return np.linalg.norm(encodings[0] - vector[0])

    items = []
    for i,j in zip(code_times,encodings):
        items.append([j,i])

    items = sorted(items,key=dist)
    print time.time() - start
    for item in items:
        print item[1],dist(item)

    return jsonify(json_obj)

if __name__ == "__main__":
    # connect to ip adress
    app.run(host='0.0.0.0')
