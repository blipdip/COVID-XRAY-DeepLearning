#Initialisation and Declarations
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop,SGD,Adam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image
sys.modules['Image'] = Image 
import cv2
tf.compat.v1.set_random_seed(2019)

rows, cols = 180,180
DATADIR_tr = "C:/Users/retr0/Downloads/hist/Train/"
DATADIR_val = "C:/Users/retr0/Downloads/hist/Validation"
input_shape = (rows,cols,1)
bs=30
epochs=30

#Pre-Processing and Importing
train = ImageDataGenerator(rescale= 1/255)
validation = ImageDataGenerator(rescale= 1/255)
test = ImageDataGenerator(rescale= 1/255)
train_dataset = train.flow_from_directory(DATADIR_tr,target_size=(180,180), batch_size=30, class_mode = 'categorical', color_mode='grayscale')
validation_dataset = validation.flow_from_directory(DATADIR_val,target_size=(180,180), batch_size=30, class_mode = 'categorical', color_mode='grayscale')
test_dataset = test.flow_from_directory("C:/Users/retr0/Downloads/hist/Test",target_size=(180,180), batch_size=30, class_mode = 'categorical', color_mode='grayscale')

#Model Definition
model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation = "relu" , input_shape = input_shape) ,
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation = "relu") ,  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation = "relu") ,  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation = "relu"),  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(550,activation="relu"),      #Hidden layer
    tf.keras.layers.Dropout(0.1,seed = 2019),
    tf.keras.layers.Dense(400,activation ="relu"),
    tf.keras.layers.Dropout(0.3,seed = 2019),
    tf.keras.layers.Dense(300,activation="relu"),
    tf.keras.layers.Dropout(0.4,seed = 2019),
    tf.keras.layers.Dense(200,activation ="relu"),
    tf.keras.layers.Dropout(0.2,seed = 2019),
    tf.keras.layers.Dense(3,activation = "softmax")   #Output Layer
])

#Model Compilation
adam=Adam(learning_rate=0.001)
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model2.summary() #view compiled model summary

#Model Training
model2_fit = model2.fit(train_dataset,
                       validation_data=validation_dataset,
                       steps_per_epoch=150 // bs,
                       epochs=epochs,
                       validation_steps=50 // bs,
                       verbose=2)

#Model Evaluation
model2.evaluate(test_dataset)