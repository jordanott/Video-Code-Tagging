from flask import Flask,request,jsonify
from keras.preprocessing.image import load_img
import random
import sys
import os
app = Flask(__name__)
# scp ott109@192.168.200.30:/home/ott109/Video-Code-Tagging/Models/CNN/ae.h5 .
sys.path.append('../Models/CNN/')
from model import *

@app.route('/')
def hello_world():
    return 'Hello, World!'

def pull_video(url,quality):
    # generate random num for directory name
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
    json_obj = {
    'youtubeId':request.args['v'],
    'code_times':[]
    }
    link += '?'
    # build link from args
    for key in request.args.keys():
        link += key + '=' + request.args[key]
    success = False
    qualities = ['_720','_480','_360']
    error = 0
    start = time.time()
    # pull video from YouTube
    while not success:
        directory = pull_video(link,qualities[error])
        if directory:
            success = True
            break
        error += 1
    print 'Time to pull video',time.time() - start
    if error == len(qualities):
        return False
    # load model and weights
    model = VGG((300,300,3),2)
    model.load_weights('../Fold_0/code_vs_no_code_strict.h5')
    # load images from new video
    images = np.empty((1,300,300,3))
    for img in os.listdir(directory):
        if img.endswith('png'):
            image = np.array(load_img(directory+img,target_size=(300,300,3))).reshape(1,300,300,3)
            images = np.append(images,image,axis=0)
    # predict images
    start = time.time()
    predictions = model.predict(images)
    print 'Predicting images time', time.time() - start
    print predictions.shape
    # get indexes where it was predicted code
    code_times = np.squeeze(np.array(np.where(np.argmax(predictions,axis=1)==0)))
    # get images that were predicted code
    code_images = images[code_times]
    # create time stamps and store in json_obj
    json_obj['code_times'] = make_timestamps(code_times)
    # load autoencoder
    encoder = conv_e((300,300,3))
    autoencoder = conv_ae((300,300,3))
    autoencoder.load_weights('../Models/CNN/ae.h5')
    # load autoencoder pretrained weights into encoder network
    for i in range(7):
        params = autoencoder.layers[i].get_weights()
        if params != []:
            encoder.layers[i].set_weights([params[0],params[1]])
    # predict encodings
    start = time.time()
    encodings = encoder.predict(code_images)
    print 'Encode time', time.time() - start
    print encodings.shape
    # store [encoding,video-time] in items
    for i,j in zip(code_times,encodings):
        items.append([j,i])

    time_steps = {}
    tmp = []
    # compare encodings to each other
    start = time.time()
    for i in range(len(items)):
        tuples = np.empty((1,2))
        for j in range(len(items)):
            # euclidean distance between encodings
            dist = np.linalg.norm(items[i][0] - items[j][0])
            tmp.append(dist)
            # add distance and index 
            tuples = np.append(tuples,np.array([dist,items[j][1]]))
        # store all distances for a given video time
        time_steps[items[i][1]] = tuples
    print 'Comparing encodings', time.time() - start
    print encodings.shape

    std = np.std(tmp)
    mean = np.mean(tmp)
    for key in time_steps.keys():
        where = np.where(time_steps[key][:,0] < mean - 3*std)
        time_steps[key] = time_steps[key][where][:,1]

    json_obj['similar_times'] = time_steps
    return jsonify(json_obj)

if __name__ == "__main__":
    # connect to ip adress
    app.run(host='0.0.0.0')
