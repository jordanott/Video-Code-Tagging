# Format of links.txt

We are interested in YouTube videos containing code tutorials. We need videos specifically marked with the time frame code is visable on the screen and the time frames when it is anything but code. Try to obtain roughly half of each type. For now we are interested in videos specifically on the Java programming language.

I have taken care of automating the video download process and splitting the videos into images.

#### links.txt should be in the following format
The separation of elements with a ```' | '``` is essential  
URL | VideoName |  CS<sub>1</sub>-CE<sub>1</sub>,CS<sub>2</sub>-CE<sub>2</sub>,...,CS<sub>n</sub>-CE<sub>n</sub> | NS<sub>1</sub>-NE<sub>1</sub>,NS<sub>2</sub>-NE<sub>2</sub>,...,NS<sub>n</sub>-NE<sub>n</sub> 

The below terms refer to time segments in the video regarding the presence of code on the screen.

CS<sub>i</sub>: Code start time
  * time when code appears in the frame  
  * one second after NE<sub>i-1</sub> 

CE<sub>i</sub>: Code End time  
  * time just before code is no longer in the frame  

NS<sub>i</sub>: Non-code start time
  * time when something non-code related appears in the frame  
  * one second after CE<sub>i-1</sub>

NE<sub>i</sub>: Non-code end time  
  * time just code is in the video frame/end of video  

