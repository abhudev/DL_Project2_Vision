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
        # print(tf.reduce_max(inputImg))      
        o1 = lrn(self.pool1(self.conv1(inputImg)), 2, 2e-05, 0.75)
        o2 = lrn((self.pool2(self.conv2(self.pad(o1)))), 2, 2e-05, 0.75)
        o3 = self.conv3(o2)
        o4 = self.conv4(o3)
        o5 = self.conv5(o4)
        # print(tf.shape(o5))
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

# Move pooling layers after conv layers
class AttnAlexnet(tf.keras.Model):
    def __init__(self, num_classes, keep_prob, cost='dp', combine='concat'):
        super(BaseAlexnet, self).__init__()
        # Possibly experiment - different initializations
        # TODO - regularization? see paper
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.cost = cost
        self.combine = combine
        self.conv1 = Conv2D(96, 11, strides=(4,4), activation='relu')
        self.pad = ZeroPadding2D(padding=(2,2))
        self.conv2 = Conv2D(256, 5, activation='relu')
        self.conv3 = Conv2D(384, 3, padding='same', activation='relu')
        self.conv4 = Conv2D(384, 3, padding='same', activation='relu')
        self.conv5 = Conv2D(256, 3, padding='same', activation='relu')
        self.pool1 = MaxPool2D(pool_size=(3,3), strides=(2,2))
        self.pool2 = MaxPool2D(pool_size=(3,3), strides=(2,2))
        self.pool3 = MaxPool2D(pool_size=(3,3), strides=(2,2))
                
        self.flat = Flatten()
        self.fc6 = Dense(units=4096, activation='relu')
        self.drop6 = Dropout(rate=self.keep_prob)
        self.fc7 = Dense(units=4096, activation='relu')
        
        self.linear_map4 = Dense(units=384, use_bias=False)
        self.linear_map5 = Dense(units=256, use_bias=False)
        if(self.cost == 'pc'):
            self.u4 = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[1,1,1,384]))
            self.u5 = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[1,1,1,256]))
        
        if(self.combine == 'concat'):
            self.attn_fc = Dense(units=num_classes, use_bias=False)
        elif(self.combine == 'indep'):
            self.attn_fc4 = Dense(units=num_classes, use_bias=False)
            self.attn_fc5 = Dense(units=num_classes, use_bias=False)


        self.drop7 = Dropout(rate=self.keep_prob)
        self.fc8 = Dense(units=self.num_classes)
        
        
    # Input - datum[0], datum[1] or datum[2], datum[3]
    def call(self, inputImg, mode='train'):  
        # print(tf.reduce_max(inputImg))      
        o1 = lrn(self.conv1(inputImg), 2, 2e-05, 0.75)
        o2 = lrn(self.conv2(self.pad(o1)), 2, 2e-05, 0.75)
        o3 = self.conv3(o2)
        o4 = self.conv4(o3)
        o5 = self.conv5(o4)
        # Shifted pooling layers down
        pooled_o5 = self.pool3(self.pool2(self.pool1(o5)))
        flat_o5 = self.flat(pooled_o5)
        o6 = self.fc6(flat_o5)
        if(mode == 'train'):
            o6 = self.drop6(o6)
        o7 = self.fc7(o6)
        if(mode == 'train'):
            o7 = self.drop7(o7)
        
        # Use o4 and o5 to get attention

        # Desctiption: We scale o7 to the sizes of conv4 and conv5 separately, and 

        bsize = tf.shape(o4)[0]
        numc_4, numc_5 = tf.shape(o4)[3], tf.shape(o5)[3]
        o4_x, o4_y = tf.shape(o4)[1].numpy(), tf.shape(o4)[2].numpy()
        o5_x, o5_y = tf.shape(o5)[1].numpy(), tf.shape(o5)[2].numpy()

        #-----------------------------------------------
        # Optional - map o4, o5 UP
        # upsample_4, upsample_5 = self.linear_map4(o4), self.linear_map5(o5)
        # o7 = tf.reshape(o7, [bsize, 1, 1, 4096])
        # c_4, c_5 = [], []
        # if(self.cost == 'dp'):
        #     c_4, c_5 = tf.reduce_sum(upsample_4 * o7, axis=-1), tf.reduce_sum(upsample_5 * o7, axis=-1)
        # elif(self.cost == 'pc'):
        #     c_4, c_5 = tf.reduce_sum(self.u4*(upsample_4+o7), axis=-1), tf.reduce_sum(self.u5*(upsample_5+o7), axis=-1)
        #-----------------------------------------------

        # reshaped_o4, reshaped_o5 = tf.reshape(o4, [-1, numc_4]), tf.reshape(o5, [-1, numc_5])
        map_to_4, map_to_5 = self.linear_map4(o7), self.linear_map5(o7)
        map_to_4, map_to_5 = tf.reshape(map_to_4, [bsize, 1, 1, numc_4]), tf.reshape(map_to_5, [bsize, 1, 1, numc_5])

        c_4, c_5 = [], []
        if(self.cost == 'dp'):
            c_4, c_5 = tf.reduce_sum(o4 * map_to_4, axis=-1), tf.reduce_sum(o5 * map_to_5, axis=-1)
        elif(self.cost == 'pc'):
            c_4, c_5 = tf.reduce_sum(self.u4*(o4+map_to_4), axis=-1), tf.reduce_sum(self.u5*(o5+map_to_5), axis=-1)
        c_4, c_5 = tf.reshape(c_4, [bsize, -1]), tf.reshape(c_5, [bsize, -1])
        a_4, a_5 = tf.nn.softmax(c_4), tf.nn.softmax(c_5)
        a_4, a_5 = tf.reshape(a_4, [bsize, o4_x*o4_y, 1]), tf.reshape(a_5, [bsize, o5_x*o5_y, 1])
        re_o4, re_o5 = tf.reshape(o4, [bsize, o4_x*o4_y, numc_4]), tf.reshape(o5, [bsize, o5_x*o5_y, numc_5])
        re_o4, re_o5 = tf.reduce_sum(re_o4 * a_4, axis=-2), tf.reduce_sum(re_o5 * a_5, axis=-2)
        
        penultimate = []
        if(self.combine == 'concat'):
            penultimate = tf.concat([re_o4, re_o5], axis=-1)
            map_out = self.attn_fc(penultimate)
            return map_out # logits
        elif(self.combine == 'indep'):
            map_out4 = self.attn_fc4(re_o4)
            map_out5 = self.attn_fc5(re_o5)
            prob_4, prob_5 = tf.nn.softmax(map_out4), tf.nn.softmax(map_out5)
            return (prob_4+prob_5)/2.0


def loss_alex(alexCNN, datum, mode):
    # Assuming datum[0] is data. datum[1] is labels    
    logits = alexCNN(datum[0], mode)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=datum[1])
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=alexCNN(datum[0], mode), labels=datum[1])
    return tf.reduce_sum(loss)/ tf.cast(tf.size(datum[1]), dtype=tf.float32)

alex_loss_grads = tfe.implicit_value_and_gradients(loss_alex)

def loss_attnalex(alexCNN, datum, mode):
    if(alexCNN.combine == 'concat'):
        # Assuming datum[0] is data. datum[1] is labels
        logits = alexCNN(datum[0], mode)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=datum[1])
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=alexCNN(datum[0], mode), labels=datum[1])
        return tf.reduce_sum(loss)/ tf.cast(tf.size(datum[1]), dtype=tf.float32)
    else:
        probs = alexCNN(datum[0], mode)
        loss = probs[datum[1]] # Probability of the CORRECT label!
        return tf.reduce_sum(loss)/ tf.cast(tf.size(datum[1]), dtype=tf.float32)

attnalex_loss_grads = tfe.implicit_value_and_gradients(loss_attnalex)

