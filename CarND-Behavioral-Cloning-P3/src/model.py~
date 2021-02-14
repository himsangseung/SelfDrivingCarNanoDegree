import csv
import cv2
import numpy as np

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

steerCorrection = 0.2
w = 320#70
h = 160#25

w_crop = 70
h_crop = 25

cnt = 0
images = []
measurements = []
for line in lines:
    measurement = float(line[3]) 

    for i in range(3):
        fullPath = '../data/IMG/'+ line[i].split('/')[-1]
        image = cv2.imread(fullPath)
        if np.shape(image) == ():
            continue
        image = cv2.resize(image, dsize=(w, h))

        if i == 1: # left camera
            measurement += steerCorrection
        elif i == 2: # right camera
            measurement -= steerCorrection
        cnt += 1
    #if cnt > 1000:
    #    break
    # image = np.expand_dims(image, 0) 
    measurements.append(measurement)
    images.append(image)
    
# Data Augmentation

augImages = images
augMeasurements = measurements
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(cv2.flip(image,1))       
  augmented_measurements.append(measurement *-1.0)

X_train = np.array(augmented_images)
Y_train = np.array(augmented_measurements)

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Lambda, Flatten,Dense, Cropping2D, Dropout
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x /255.0 -0.5,  
          input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70, 25),(0,0))))
# Testing
#model.add(Flatten())#w,h,3)))
#model.add(Dense(1))
#model.compile(loss='mse', optimizer='adam')

# Lenet
#model.add(Conv2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D(padding='same'))
#model.add(Conv2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D(padding='same'))
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))

# NVIDIA
model.add(Conv2D(24,5,2,activation="relu", padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(36,5,2,activation="relu",padding='same'))
model.add(Dropout(0.2))
#model.add(MaxPooling2D(padding='same'))
model.add(Conv2D(48,5,2,activation="relu",padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64,3,activation="relu",padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64,3,activation="relu",padding='same'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
object_history = model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True, epochs=10)

### print the keys contained in the history object
print(object_history.history.keys())

import matplotlib.pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(object_history.history['loss'])
plt.plot(object_history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
plt.savefig("Loss_train_val.png", bbox_inches='tight')


model.save('model.h5')
