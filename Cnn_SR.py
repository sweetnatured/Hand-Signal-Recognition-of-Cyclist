import tensorflow as tf
import os
import random
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import cv2

DATADIR = "/home/salih/Desktop/Signals/Train"
CATEGORIES_TRAIN=["Train_leftCross","Train_leftSignal","Train_leftVert","Train_rightCross","Train_rightSignal","Train_rightVert"]


training_data=[]

def create_training_data():

    for category in CATEGORIES_TRAIN:
    
         path=os.path.join(DATADIR,category)
         class_number=CATEGORIES_TRAIN.index(category)
         

         for img in os.listdir(path):
             
             try:
                  
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(50,50))
                norm_image = cv2.normalize(new_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                training_data.append([norm_image,class_number])

             except Exception as e:

                pass
            


DATADIR_TEST = "/home/salih/Desktop/Signals/Test"
CATEGORIES_TRAIN_TEST=["Test_leftCross","Test_leftSignal","Test_leftVert","Test_rightCross","Test_rightSignal","Test_rightVert"]


testing_data=[]

def create_testing_data():

    for category in CATEGORIES_TRAIN_TEST:
    
         path=os.path.join(DATADIR_TEST,category)
         class_number=CATEGORIES_TRAIN_TEST.index(category)

         for img in os.listdir(path):

             try:
                  
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                
                new_array = cv2.resize(img_array,(50,50))
                norm_image = cv2.normalize(new_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                testing_data.append([norm_image,class_number])


             except Exception as e:

                pass


create_training_data()
random.shuffle(training_data)

create_testing_data()
random.shuffle(testing_data)

Features_train=[]
Labels_train=[]

for features,labels in training_data:

   Features_train.append(features)
   Labels_train.append(labels)

I=np.array(Features_train).reshape(-1,50,50,1)



Features_test=[]
Labels_test=[]

for features,labels in testing_data:

   Features_test.append(features)
   Labels_test.append(labels)

T=np.array(Features_test).reshape(-1,50,50,1)




model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50,50,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(I, Labels_train, epochs=10, 
                    validation_data=(T, Labels_test))




test_loss, test_acc = model.evaluate(T,  Labels_test, verbose=2)  

print(model.predict(T[:5]))
print(Labels_test[:5])


print(test_acc)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

model.save('my_model.h5')







