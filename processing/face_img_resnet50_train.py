# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:03:20 2022

@author: mehmoodyar.baig
"""
# import generic libraries

import matplotlib.pyplot as plt
import pickle

# import deep learning libraries
from keras.layers import Dense, Flatten
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# set path to data
data_dir = 'dataset/img/'

img_height, img_width = 180, 180
batch_size = 128
epochs = 5

# splitting the data set
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset='training',
    seed=132,
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset='validation',
    seed=132,
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

# Callback to keep control of validation loss
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss", patience=3, verbose=True)

# training the model
resnet_model = Sequential();

pretrained_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling='avg',
    classes=1000)

for layer in pretrained_model.layers:
    layer.trainable = False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(128, activation='relu'))
resnet_model.add(Dense(64, activation='relu'))
resnet_model.add(Dense(10, activation=('softmax')))
resnet_model.add(Dense(2, activation=('sigmoid')))

resnet_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = resnet_model.fit(train_ds,
                           validation_data=val_ds,
                           epochs=epochs, callbacks=early_stop, use_multiprocessing=True)

# #Evaluating the model
# fig1 = plt.gcf()
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.axis(ymin = 0.4, ymax = 1)
# plt.grid()
# plt.title('Model Accuracy')

# #loss
# plt.plot(history.history['loss'], label = 'train loss')
# plt.plot(history.history['val_loss' ], label = 'Validation loss')
# plt.grid()
# plt.title('Model Loss')

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.axis(xmin=0, xmax=epochs, ymin=0, ymax=1.2)
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("No. of Epochs ")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

#saving the model to disk
pickle.dump(resnet_model, open('model.pkl','wb'))


