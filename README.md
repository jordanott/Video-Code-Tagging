# Video Code Tagging

This repository contains work for two projects. The first is interested in identifying the presence of Java code in software engineering video tutorials. The second is interested in predicting Java versus Python code in image frames.

  ⇒ See [Labeling Standards](DataCollection/LABELING.md) for info on how we label data    
  ⇒ See [DataCollection](DataCollection/) for how we aquire data  
  ⇒ See [Models](Models/) for the networks we use to label images   
  ⇒ See [Results](Results/) for results on identifying Java code in videos
  ⇒ See [Java Python Results](jp_Results/) for results of discriminating between Java and Python code in videos
  ⇒ See [Server](Server/) how our tagging tool works on the backend

### Steps ###
1. Acquire links of videos  
  * Run ```bash DataCollection/grab.sh``` to pull all the videos and split them into frames  
2. Label data  
  * Ex: Image that contains Java code
    * path/to/img,1,0,0,0
3. Resize data set  
  * Run ```mv data_preprocess.sh path/where/dataset/is```  
  * Run ```bash data_preprocess.sh``` resize all images in dataset  
4. Train your model
  * Run ```python run.py```  
    * This loads training set with labels and begins training
