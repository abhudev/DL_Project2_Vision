import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import os
import time
from tensorflow.keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, Dense, Dropout, Flatten
from tensorflow.nn import local_response_normalization as lrn

# Followed tutorial on https://github.com/madalinabuzau/tensorflow-eager-tutorials/blob/master/07_convolutional_neural_networks_for_emotion_recognition.ipynb

class BaseAlexnet(tf.keras.Model):
    def __init__(self, num_classes, keep_prob):
        super(BaseAlexnet, self).__init__()
        # Possibly experiment - different initializations
        # TODO - regularization? see paper
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.conv1 = Conv2D(96, 11, strides=(4,4), activation='relu')
        self.pool1 = MaxPool2D(pool_size=(3,3), strides=(2,2))
        self.pad = ZeroPadding2D(padding=(2,2))
        self.conv2 = Conv2D(256, 5, activation='relu')
        self.pool2 = MaxPool2D(pool_size=(3,3), strides=(2,2))
        self.conv3 = Conv2D(384, 3, padding='same', activation='relu')
        self.conv4 = Conv2D(384, 3, padding='same', activation='relu')
        self.conv5 = Conv2D(256, 3, padding='same', activation='relu')
        self.pool3 = MaxPool2D(pool_size=(3,3), strides=(2,2))
        self.flat = Flatten()
        self.fc6 = Dense(units=4096, activation='relu')
        self.drop6 = Dropout(rate=self.keep_prob)
        self.fc7 = Dense(units=4096, activation='relu')
        self.drop7 = Dropout(rate=self.keep_prob)
        self.fc8 = Dense(units=self.num_classes)
        
        
    # Input - datum[0], datum[1] or datum[2], datum[3]
    def call(self, inputImg, mode='train'):        
        o1 = lrn(self.pool1(self.conv1(inputImg)), 2, 2e-05, 0.75)
        o2 = lrn((self.pool2(self.conv2(self.pad(o1)))), 2, 2e-05, 0.75)
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
        logits = self.fc8(o7)
        return logits 
        # Return 

# class AttnAlexnet(tf.keras.Model):
#     def __init__(self, num_classes, keep_prob):
#         super(AttnAlexnet, self).__init__()
#         # Possibly experiment - different initializations
#         # TODO - regularization? see paper

def loss_alex(alexCNN, datum, mode):
    # Assuming datum[0] is data. datum[1] is labels
    logits = alexCNN(datum[0], mode)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=datum[1])
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=alexCNN(datum[0], mode), labels=datum[1])
    return tf.reduce_sum(loss)/ tf.cast(tf.size(datum[1]), dtype=tf.float32)

alex_loss_grads = tfe.implicit_value_and_gradients(loss_alex)


