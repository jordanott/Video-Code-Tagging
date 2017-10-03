# Format of links.txt

We are interested in YouTube videos containing code tutorials. We need videos specifically marked with the time frame code is visable on the screen and the time frames when it is anything but code. Try to obtain roughly half of each type. For now we are interested in videos specifically on the Java programming language. It is important to get a variety of videos that use different size font, text editors, colors, etc. We will be training convolutional networks on these images so variety in the data is very important!

I have taken care of automating the video download process and splitting the videos into images.

### Requirements ###  
Gather data using [PyTube](https://github.com/nficano/pytube)
```
pip install pytube
sudo apt-get install ffmpeg
```
###### links.txt Format ######
Link | Video Name

#### To Run ####
```
# create tmux session
tmux new -s grab_videos
bash grab.sh
```
