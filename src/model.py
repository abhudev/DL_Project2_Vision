import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import os
import time
from tf.keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, Dense, Dropout, Flatten


# Followed tutorial on https://github.com/madalinabuzau/tensorflow-eager-tutorials/blob/master/07_convolutional_neural_networks_for_emotion_recognition.ipynb

class Alex_attn(tf.keras.Model):
    def __init__(self, num_classes, keep_prob):
        super(Alex_attn, self).__init__()
        # Possibly experiment - different initializations
        # TODO - regularization? see paper
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.conv1 = Conv2D(96, 11, strides=(4,4), activation='relu')
        self.pool1 = MaxPool2D(pool_size=(3,3), strides=(2,2))
        self.norm1 = "Do this later"
        self.pad = ZeroPadding2D(padding=(2,2))
        self.conv2 = Conv2D(256, 5,, activation='relu')
        self.pool2 = MaxPool2D(pool_size=(3,3), strides=(2,2))
        self.norm2 = "Do this later"
        self.conv3 = Conv2D(384, 3, padding='same')
        self.conv4 = Conv2D(384, 3, padding='same')
        self.conv5 = Conv2D(256, 3, padding='same')
        self.pool3 = MaxPool2D(pool_size=(3,3), strides=(2,2))
        self.flat = Flatten()
        self.fc6 = Dense(units=4096)
        self.drop6 = Dropout(rate=self.keep_prob)
        self.fc7 = Dense(units=4096)
        self.drop7 = Dropout(rate=self.keep_prob)
        self.fc8 = Dense(units=self.num_classes)
        
        
    # Input - datum[0], datum[1] or datum[2], datum[3]
    def call(self, input, mode='train'):        
        o1 = self.norm1(self.pool1(self.conv1(input)))
        o2 = self.norm2(self.pool2(self.conv2(self.pad(o1))))
        o3 = self.conv3(o2)
        o4 = self.conv4(o3)
        o5 = self.conv5(o4)
        pooled_o5 = self.pool3(o5)
        flat_o5 = self.flat(pooled_o5)
        o6 = self.fc6(flat_o5)
        if(mode == 'train'):
            o6 = self.drop6(o6)
        o7 = self.fc7(o6)
        if(mode == 'train'):
            o7 = self.drop7(o7)
        o8 = self.fc8(o7)
        # Return 
