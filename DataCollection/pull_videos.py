import os
import random

os.mkdir('Downloads')

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
        os.system('rm -rf '+path)
        return False
    video = '{path}/{name}.mp4'.format(path=path,name=name)
    # split video into images
    if os.system('ffmpeg -i "{video}" -r 1 -f image2 "{path}"/%d.png -nostdin'.format(video=video,path=path)):
        return False
    return path

links = open('python_links.txt')
for link in links:
    success = False
    qualities = ['_720','_480','_360']
    error = 0
    # pull video from YouTube
    while not success:
        directory = pull_video(link,qualities[error])
        if directory:
            success = True
            break
        error += 1
