import os
import csv
from random import shuffle

samples = []
#Opening csv file:
with open('test1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
#splitting training data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
              name = 'test1/IMG/'+batch_sample[0].split('/')[-1]
              center_image = cv2.imread(name)
              #converting image from BGR to RGB
	            center_image = cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB)
              center_angle = float(batch_sample[3])

              #Flipping the image
              flip=cv2.flip(center_image,1)
              images.append(center_image)
              angles.append(center_angle)
              images.append(flip)
              angles.append(center_angle*-1)
            
              left_image=cv2.imread('test1/IMG/'+batch_sample[1].split('/')[-1])
              left_image=cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB)
              # left_flip=cv2.flip(left_image,1)
               # left_image=left_image[50:140,:]
              left_angle=center_angle+0.2
              right_image=cv2.imread('test1/IMG/'+batch_sample[2].split('/')[-1])
              right_image=cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)
              # right_flip=cv2.flip(right_image,1)
              right_angle=center_angle-0.2
              images.append(left_image)
              images.append(right_image)
              angles.append(left_angle)
              angles.append(right_angle)
              # images.append(left_flip)
              # images.append(right_flip)
              # angles.append(left_angle*-1)
              # angles.append(right_angle*-1)

            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def createPreProcessingLayers():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model
def nVidiaModel():
    """
    Creates nVidea Autonomous Car Group model
    """
    model = createPreProcessingLayers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu',W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu',W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu',W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64,3,3, activation='relu',W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64,3,3, activation='relu',W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(100,W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(50,W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(10,W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(1))
    return model

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten,Dense,Activation,Dropout,Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras import regularizers
model= nVidiaModel()

model.compile(loss='mse', optimizer='adam',metric='accuracy')
model.fit_generator(train_generator, samples_per_epoch=
                    len(train_samples)*4, validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=10
,verbose=1,show_accuracy=True)

#Saving the mddel.
model.save('model.h5')
