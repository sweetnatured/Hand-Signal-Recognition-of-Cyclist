from darkflow.net.build import TFNet 
import cv2
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random



new_model = tf.keras.models.load_model('my_model.h5')


def numbers_to_strings(argument): 
    switcher = { 
        0: "leftCross", 
        1: "leftSignal", 
        2: "leftVert", 
        3: "rightCross", 
        4: "rightSignal",  
        5: "rightVert" , 

      } 
    return switcher.get(argument, "nothing")


def boxing(original_img, predictions,output,height):
    newImage = np.copy(original_img)
    otp = numbers_to_strings(output)    

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.3:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
            
    newImage = cv2.putText(newImage, otp, (350,500), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 0,230), 2, cv2.LINE_AA)  
            
    return newImage


def find_bom(lst):     # finds bottom of bicyle or motorbike

   for i in range(len(lst)):
          
        if(lst[i]['label']=="motorbike" or lst[i]['label']=="bicycle"):
                 
               return (lst[i]['topleft']['y']+lst[i]['bottomright']['y'])/2 , (lst[i]['topleft']['x']+lst[i]['bottomright']['x'])/2 


def find_persons(lst):
   persons=[]
   for i in range(len(lst)):

      if(lst[i]['label']=="person"):
      
             persons.append(lst[i])          

   return persons

def find_mid(lst):
   
   liste_mid=[]
   
   for i in range(len(lst)):

      
     if(lst[i]['label']=="person"):

         middles= (lst[i]['topleft']['y']+lst[i]['bottomright']['y'])/2 , (lst[i]['topleft']['x']+lst[i]['bottomright']['x'])/2
         liste_mid.append(middles) 

   return liste_mid


def find_dist(bicy,prsns):

   distances=[]
   
   for prsn in prsns:
    
     dist=np.sqrt(np.square(bicy[0]-prsn[0])+np.square(bicy[1]-prsn[1]))
     distances.append(dist)


   return distances.index(min(distances))


def crop_details(lst,indx):

   return lst[indx]['topleft'],lst[indx]['bottomright']

  

def crop_image(img,crdnts):

   left_upper_x=crdnts[0]['x']
   left_upper_y=crdnts[0]['y']
   right_bottom_x=crdnts[1]['x']
   right_bottom_y=crdnts[1]['y']
   return img[left_upper_y:right_bottom_y , left_upper_x:right_bottom_x]



options = {"model": "cfg/yolo.cfg", 
           "load": "bin/yolov2.weights", 
           "threshold": 0.4, 
           "gpu": 0}

tfnet = TFNet(options)   


cap = cv2.VideoCapture('onden.avi')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (int(width), int(height)))




while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == True:
        
        frame = np.asarray(frame)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        results = tfnet.return_predict(frame)
        
        x=find_bom(results)
        prsns=find_persons(results)
        try:
         if(len(prsns)>1):

           y=find_mid(prsns)
           z=find_dist(x,y)
           t=crop_details(prsns,z)
           new_frame=crop_image(frame,t)

         else:

           m=crop_details(prsns,0)
           new_frame=crop_image(frame,m)
        
        except:
          pass

        new_fram=cv2.cvtColor(new_frame,cv2.COLOR_BGR2GRAY)
        new_array = cv2.resize(new_fram,(50,50))
        norm_image = cv2.normalize(new_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        print("selam")
        T=np.array(norm_image).reshape(-1,50,50,1)
        print(np.argmax(new_model.predict(T)))
        print(height,width)

        new_frames = boxing(frame, results,np.argmax(new_model.predict(T)),height)
        
        # Display the resulting frame
        out.write(new_frames)
        cv2.imshow('frame',new_frames)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
            break


cap.release()
out.release()
cv2.destroyAllWindows()








"""
cv2.imshow('Cropped_Frame',new_frame)
cv2.waitKey(0)
"""





"""
new_frame = boxing(img, results)
cv2.imshow('frame',new_frame)
cv2.waitKey(0)
"""
