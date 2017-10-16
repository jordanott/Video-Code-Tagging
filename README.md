# Video Code Tagging

We are interested in creating a deep learning model capable of tagging video frames that contain coding sections, specifically in programming tutorials. We are interested in frames that contain the Java programming language.

  ⇒ See [Labeling Standards](DataCollection/LABELING.md) for info on how we label data    
  ⇒ See [DataCollection](DataCollection/) for how we aquire data  
  ⇒ See [Models](Models/) for the networks we use to label images   

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
