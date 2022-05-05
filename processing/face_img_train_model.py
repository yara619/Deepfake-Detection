# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:03:20 2022

@author: mehmoodyar.baig
"""
#import generic libraries 
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

#import deep learning libraries  
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from keras.models import Sequential

CNN_Classifier=Sequential();

# Step 1 - Convolution_1
CNN_Classifier.add(Conv2D(40,(3,3),input_shape=(64,64,3),activation='relu'))

# Step 2 - Pooling_1
CNN_Classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 1 - Convolution_2
CNN_Classifier.add(Conv2D(12,(3,3),activation='relu'))

# Step 2 - Pooling_2
CNN_Classifier.add(MaxPooling2D(pool_size=(2,2)))


# Step 3 - Flattening
CNN_Classifier.add(Flatten())


# Step 4 - Full connection

CNN_Classifier.add(Dense(units=140, activation='relu'))

# Adding the First Hidden Layer
CNN_Classifier.add(Dense(units=40, activation='relu'))

# Adding the Second Hidden Layer
CNN_Classifier.add(Dense(units=25, activation='relu'))

# Adding the Output Layer
CNN_Classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
CNN_Classifier.compile(optimizer ='adam',loss='binary_crossentropy',metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/images/train_data',
                                                 target_size = (64, 64),
                                                 batch_size = 64,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/images/test_data',
                                            target_size = (64, 64),
                                            batch_size = 62,
                                            class_mode = 'binary')

CNN_Classifier.fit_generator(training_set,
                         steps_per_epoch = 70,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 50)

# Prediction for dog/cat image using train model

from tensorflow.keras.preprocessing import image
test_image = image.load_img ('dataset/images/single_prediction/1814_fake.jpg', target_size= (64, 64))
test_image.show()
test_image = image.img_to_array (test_image)
test_image = np.expand_dims(test_image, axis = 0)
results = CNN_Classifier.predict(test_image)
training_set.class_indices

print (results)

if results[0][0] == 1:
    prediction = 'real'
    print(prediction)
else:
    prediction = 'fake'
    print(prediction)