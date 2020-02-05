import Augmentor 
import os
# Passing the path of the image directory 

DATADIR = "/home/salih/Desktop/Signals"
CATEGORIES=["leftSignal","rightSignal","rightCross","leftCross","rightVert","leftVert"]

for category in CATEGORIES:
    
     path=os.path.join(DATADIR, category)
     p = Augmentor.Pipeline(path) 
     p.black_and_white(0.1) 
     p.rotate(0.3, 10, 10) 
     p.skew(0.4, 0.5) 
     p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.5) 
     p.sample(150) 






