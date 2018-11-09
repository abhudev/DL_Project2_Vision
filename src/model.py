import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import os
import time
from tensorflow.keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, Dense, Dropout, Flatten, Add, GlobalAveragePooling2D, BatchNormalization
from tensorflow.nn import local_response_normalization as lrn

# Followed tutorial on https://github.com/madalinabuzau/tensorflow-eager-tutorials/blob/master/07_convolutional_neural_networks_for_emotion_recognition.ipynb

#------------------------------Base Alexnet------------------------------#
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
        

    def update_last_layer(self, num_classes):
        self.num_classes = num_classes
        self.fc8 = Dense(units=num_classes)

    # Input - datum[0], datum[1] or datum[2], datum[3]
    def call(self, inputImg, mode='train'):  
        # print(tf.reduce_max(inputImg))      
        # print(tf.shape(inputImg))
        o1 = lrn(self.pool1(self.conv1(inputImg)), 2, 2e-05, 0.75)
        o2 = lrn((self.pool2(self.conv2(self.pad(o1)))), 2, 2e-05, 0.75)
        o3 = self.conv3(o2)
        o4 = self.conv4(o3)
        o5 = self.conv5(o4)
        # print(tf.shape(o5))
        pooled_o5 = self.pool3(o5)
        flat_o5 = self.flat(pooled_o5)
        o6 = self.fc6(flat_o5)
        # if(mode == 'train'):
        #     o6 = self.drop6(o6)
        o7 = self.fc7(o6)
        # if(mode == 'train'):
        #     o7 = self.drop7(o7)
        logits = self.fc8(o7)
        return logits 
        # Return 

def loss_alex(alexCNN, datum, mode):
    # Assuming datum[0] is data. datum[1] is labels    
    logits = alexCNN(datum[0], mode)
    # print(tf.shape(datum[1]))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=datum[1])
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=alexCNN(datum[0], mode), labels=datum[1])
    return tf.reduce_sum(loss)/ tf.cast(tf.size(datum[1]), dtype=tf.float32)

alex_loss_grads = tfe.implicit_value_and_gradients(loss_alex)

#------------------------------Base Alexnet------------------------------#



#------------------------------Attention Alexnet------------------------------#
class AttnAlexnet(tf.keras.Model):
    def __init__(self, num_classes, keep_prob, cost='dp', combine='concat', sample='down'):
        super(AttnAlexnet, self).__init__()
        # Possibly experiment - different initializations
        # TODO - regularization? see paper
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.cost = cost
        self.combine = combine
        self.sample = sample

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
        # NOTE - check - use bias ok?
        self.linear_map4 = Dense(units=384, use_bias=False)
        self.linear_map5 = Dense(units=256, use_bias=False)
        self.o4_to_out = Dense(units=4096, use_bias=False)
        self.o5_to_out = Dense(units=4096, use_bias=False)
        if(self.cost == 'pc'):
            if(self.sample == 'down'):
                self.u4 = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[1,1,384]))
                self.u5 = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[1,1,256]))
            elif(self.sample == 'up'):
                self.u4 = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[1,1,4096]))
                self.u5 = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[1,1,4096]))
        
        if(self.combine == 'concat'):
            self.attn_fc = Dense(units=num_classes, use_bias=False)
        elif(self.combine == 'indep'):
            self.attn_fc4 = Dense(units=num_classes, use_bias=False)
            self.attn_fc5 = Dense(units=num_classes, use_bias=False)


        self.drop7 = Dropout(rate=self.keep_prob)
        self.fc8 = Dense(units=self.num_classes)

        self.map_4_to_5 = Dense(units=256)
        self.add_4_5 = Add()
        
    def update_last_layer(self, num_classes):
        self.num_classes = num_classes
        self.fc8 = Dense(units=self.num_classes)
        if(self.combine == 'concat'):
            self.attn_fc = Dense(units=num_classes, use_bias=False)
        elif(self.combine == 'indep'):
            self.attn_fc4 = Dense(units=num_classes, use_bias=False)
            self.attn_fc5 = Dense(units=num_classes, use_bias=False)
        
    # Input - datum[0], datum[1] or datum[2], datum[3]
    def call(self, inputImg, mode='train'):  
        # print(tf.reduce_max(inputImg))
        # print(inputImg.shape)
        o1 = lrn(self.pool1(self.conv1(inputImg)), 2, 2e-05, 0.75)
        # print("o1", tf.shape(o1))
        o2 = lrn(self.pool2(self.conv2(self.pad(o1))), 2, 2e-05, 0.75)
        # print("o2", tf.shape(o2))
        o3 = self.conv3(o2)
        # print("o3", tf.shape(o3))
        o3 = self.pool3(o3)
        # print("o3", tf.shape(o3))
        o4 = self.conv4(o3)
        # print("o4", tf.shape(o4))
        o5 = self.conv5(o4)
        # print("o5", tf.shape(o5))
        # Shifted pooling layers down
        # pooled_o5 = self.pool3(self.pool2(self.pool1(o5)))
        # pooled_o5 = self.pool3(o5)
        pooled_o5 = (o5)
        flat_o5 = self.flat(pooled_o5)
        # print("flat_o5", tf.shape(flat_o5))
        o6 = self.fc6(flat_o5)
        # print("o6", tf.shape(o6))
        # if(mode == 'train'):
        #     o6 = self.drop6(o6)
        o7 = self.fc7(o6)
        # print("o7", tf.shape(o7))
        # if(mode == 'train'):
        #     o7 = self.drop7(o7)
        
        # Use o4 and o5 to get attention

        # Desctiption: We scale o7 to the sizes of conv4 and conv5 separately, and 

        bsize = tf.shape(o4)[0]
        numc_4, numc_5 = tf.shape(o4)[3], tf.shape(o5)[3]
        o4_x, o4_y = tf.shape(o4)[1].numpy(), tf.shape(o4)[2].numpy()
        o5_x, o5_y = tf.shape(o5)[1].numpy(), tf.shape(o5)[2].numpy()
        
        # 1
        if(self.sample == 'down'):
            map_to_4 = self.linear_map4(o7) 
            map_to_5 = self.linear_map5(o7)
        else:
            # Alternate - map o4 and o5 up
            o4_upsample = self.o4_to_out(o4)
            o5_upsample = self.o5_to_out(o5)
        

        # 2
        if(self.sample == 'down'):
            map_to_4 = tf.expand_dims(map_to_4, axis=-1) # map_to_4 - [bsize, numc_4, 1]
            map_to_5 = tf.expand_dims(map_to_5, axis=-1) # map_to_5 - [bsize, numc_5, 1]
            # print("map_to_4", tf.shape(map_to_4))
            # print("map_to_5", tf.shape(map_to_5))
            o4 = tf.reshape(o4, [bsize, o4_x*o4_y, numc_4]) 
            o5 = tf.reshape(o5, [bsize, o5_x*o5_y, numc_5])
            # print("o4", tf.shape(o4))
            # print("o5", tf.shape(o5))

        else:
            # Alternate - sample up o4 and o5
            o4_upsample = tf.reshape(o4_upsample, [bsize, o4_x*o4_y, 4096])
            o5_upsample = tf.reshape(o5_upsample, [bsize, o5_x*o5_y, 4096])        

        # 3
        c_4 = [] 
        c_5 = []
        if(self.cost == 'dp'):
            if(self.sample == 'down'):
                c_4 = tf.matmul(o4, map_to_4)
                c_5 = tf.matmul(o5, map_to_5)
            else:
                # Alternate - upsample o4 and o5
                o7 = tf.reshape(o7, [bsize, 4096, 1])
                c_4 = tf.matmul(o4_upsample, o7)
                c_5 = tf.matmul(o5_upsample, o7)
        elif(self.cost == 'pc'):
            if(self.sample == 'down'):
                map_to_4 = tf.reshape(map_to_4, [bsize, 1, numc_4])
                map_to_5 = tf.reshape(map_to_5, [bsize, 1, numc_5])            
                c_4 = tf.reduce_sum(self.u4*(o4+map_to_4), axis=-1)
                c_5 = tf.reduce_sum(self.u5*(o5+map_to_5), axis=-1)
            else:
                # Alternative - upsample o4 and o5
                o7 = tf.reshape(o7, [bsize, 1, 4096])
                c_4 = tf.reduce_sum(self.u4*(o4_upsample+o7), axis=-1)
                c_5 = tf.reduce_sum(self.u5*(o5_upsample+o7), axis=-1)


        
        # 4
        c_4 = tf.reshape(c_4, [bsize, -1])
        c_5 = tf.reshape(c_5, [bsize, -1])
        # print("c_4", tf.shape(c_4))
        # print("c_5", tf.shape(c_5))
        a_4 = tf.nn.softmax(c_4)
        a_5 = tf.nn.softmax(c_5)        
        
        # 5
        # Shapes:
        # a_4 - [bsize, o4_x*o4_y, 1]
        # a_5 - [bsize, o5_x*o5_y, 1]
        a_4 = tf.expand_dims(a_4, axis=-1)
        a_5 = tf.expand_dims(a_5, axis=-1)
        # print("a_4", tf.shape(a_4))
        # print("a_5", tf.shape(a_5))
        o4 = tf.reshape(o4, [bsize, numc_4, o4_x*o4_y])
        o5 = tf.reshape(o5, [bsize, numc_5, o5_x*o5_y])
        # print("o4", tf.shape(o4))
        # print("o5", tf.shape(o5))

        # 6
        re_o4 = tf.matmul(o4, a_4)
        re_o5 = tf.matmul(o5, a_5)
        re_o4 = tf.squeeze(re_o4, [2])
        re_o5 = tf.squeeze(re_o5, [2])
        # print("re_o4", tf.shape(re_o4))
        # print("re_o5", tf.shape(re_o5))
        # re_o4 = self.map_4_to_5(re_o4)    # Project re_o4 to re_o5


        # penultimate = []
        
        if(self.combine == 'concat'):
            penultimate = tf.concat([re_o4, re_o5], axis=-1)
            # print("penultimate", tf.shape(penultimate))
            map_out = self.attn_fc(penultimate)
            # print("map_out", tf.shape(map_out))
            pred_class = (tf.argmax(map_out, axis=-1))
        elif(self.combine == 'indep'):
            map_out4 = self.attn_fc4(re_o4)
            map_out5 = self.attn_fc5(re_o5)
            prob_4, prob_5 = tf.nn.softmax(map_out4), tf.nn.softmax(map_out5)
            pred_class = (tf.argmax((prob_4+prob_5)/2.0, axis=-1))
        
        if(mode != 'maps'):
            if(self.combine == 'concat'):
                return map_out
            elif(self.combine == 'indep'):
                return (prob_4+prob_5)/2.0
        else:
            # Return attention maps
            # TODO - return SINGLE attention map!
            # a_4 = tf.reshape(a_4, [bsize, o4_x, o4_y])
            # a_5 = tf.reshape(a_5, [bsize, o5_x, o5_y])
            # attention_map = tf.sqrt(a_4 * a_5)
            c_4 = tf.reshape(c_4, [bsize, o4_x, o4_y])
            c_5 = tf.reshape(c_5, [bsize, o5_x, o5_y])
            
            # attention_map = c_4*c_5
            # print(attention_map[0])
            # print(c_4[0])
            # print(c_5[0])
            
            xdim, ydim = o4_x, o4_y
            c_4 = tf.reshape(c_4, [bsize, -1])
            c_5 = tf.reshape(c_4, [bsize, -1])
            
            min_val = tf.expand_dims(tf.reduce_min(c_4, axis=-1), axis=-1)
            c_4 -= min_val
            max_val = tf.expand_dims(tf.reduce_max(c_4, axis=-1), axis=-1)
            c_4 /= max_val

            min_val = tf.expand_dims(tf.reduce_min(c_5, axis=-1), axis=-1)
            c_5 -= min_val
            max_val = tf.expand_dims(tf.reduce_max(c_5, axis=-1), axis=-1)
            c_5 /= max_val

            attention_map = c_4

            # attention_map = tf.reshape(attention_map, [bsize, -1])
            # min_val = tf.expand_dims(tf.reduce_min(attention_map, axis=-1), axis=-1)
            # attention_map -= min_val
            # max_val = tf.expand_dims(tf.reduce_max(attention_map, axis=-1), axis=-1)
            # attention_map /= max_val
            attention_map = tf.reshape(attention_map, [bsize, xdim, ydim])            
            return attention_map, pred_class
            

def loss_attnalex(alexCNN, datum, mode):
    if(alexCNN.combine == 'concat'):
        # Assuming datum[0] is data. datum[1] is labels
        # logits1, logits2 = alexCNN(datum[0], mode)
        # logits = tf.concat([logits1, logits2], axis=-1)
        # logits = logits1+logits2
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=datum[1])
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=alexCNN(datum[0], mode), labels=datum[1])
        # return tf.reduce_sum(loss)/ tf.cast(tf.size(datum[1]), dtype=tf.float32)

        logits = alexCNN(datum[0], mode)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=datum[1])
        # print(tf.shape(datum[1]))
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=alexCNN(datum[0], mode), labels=datum[1])
        return tf.reduce_sum(loss)/ tf.cast(tf.size(datum[1]), dtype=tf.float32)
    elif(alexCNN.combine == 'indep'):
        probs = alexCNN(datum[0], mode)
        row_indices = tf.range(tf.size(datum[1]))
        full_indices = tf.stack([row_indices, datum[1]], axis=1)
        # print(probs)
        # print(full_indices)
        loss = tf.gather_nd(probs, full_indices) # Probability of the CORRECT label!
        # print(loss)
        return (-1.0)*tf.reduce_sum(tf.log(loss))/ tf.cast(tf.size(datum[1]), dtype=tf.float32)

attnalex_loss_grads = tfe.implicit_value_and_gradients(loss_attnalex)

#------------------------------Attention Alexnet------------------------------#



#------------------------------GAP Alexnet------------------------------#

class GAPAlexnet(tf.keras.Model):
    def __init__(self, num_classes, keep_prob):
        super(GAPAlexnet, self).__init__()
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
        self.gap = GlobalAveragePooling2D()
        # self.flat = Flatten()        
        self.drop_gap = Dropout(rate=self.keep_prob)
        self.fc8 = Dense(units=self.num_classes, use_bias=False)
        
    def update_last_layer(self, num_classes):
        self.num_classes = num_classes
        self.fc8 = Dense(units=self.num_classes, use_bias=False)
        
    # Input - datum[0], datum[1] or datum[2], datum[3]
    def call(self, inputImg, mode='train'):     
        o1 = lrn(self.pool1(self.conv1(inputImg)), 2, 2e-05, 0.75)
        o2 = lrn((self.pool2(self.conv2(self.pad(o1)))), 2, 2e-05, 0.75)
        o3 = self.conv3(o2)
        o4 = self.conv4(o3)
        o5 = self.conv5(o4)
        pooled_o5 = self.pool3(o5)
        gap_out = self.gap(pooled_o5)
        logits = self.fc8(gap_out)
        if(mode == 'train' or mode == 'eval'):
            return logits 
        elif(mode == 'maps'):
            # return tf.reduce_mean(o5, axis=-1)
            pred_class = (tf.argmax(logits, axis=-1))
            np_wts = tf.convert_to_tensor(self.fc8.get_weights()[0])
            # print("SHAPE", np_wts.shape)
            np_wts = tf.transpose(np_wts)
            # print("TRANSPOSE SHAPE", np_wts.shape)
            # weights = tf.gather(tf.transpose(tf.convert_to_tensor(self.fc8.get_weights()[0])) , pred_class)
            weights = tf.gather(np_wts , pred_class)
            # print("FINAL SHAPE", weights.shape)
            # exit()
            # weights = tf.convert_to_tensor(self.fc8.get_weights()[0][:, pred_class])
            weights = tf.expand_dims(weights, axis=-1)
            OMAP = pooled_o5
            bsize, xdim, ydim, c = tf.shape(OMAP)[0], tf.shape(OMAP)[1], tf.shape(OMAP)[2], tf.shape(OMAP)[3]
            OMAP = tf.reshape(OMAP, [bsize, xdim*ydim, c])
            cam_map = tf.squeeze(tf.matmul(OMAP, weights), [2])
            cam_map = tf.reshape(cam_map, [bsize, xdim, ydim])
            return cam_map, pred_class
        else:
            assert(False)
        # Return 

def loss_gapalex(alexCNN, datum, mode):
    # Assuming datum[0] is data. datum[1] is labels    
    logits = alexCNN(datum[0], mode)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=datum[1])
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=alexCNN(datum[0], mode), labels=datum[1])
    return tf.reduce_sum(loss)/ tf.cast(tf.size(datum[1]), dtype=tf.float32)

gapalex_loss_grads = tfe.implicit_value_and_gradients(loss_gapalex)


#------------------------------GAP Alexnet------------------------------#


#------------------------------NEW ATTENTION ALEXNET------------------------------#
class AttnAlexnet2(tf.keras.Model):
    def __init__(self, num_classes, keep_prob, cost='dp', combine='concat', sample='down'):
        super(AttnAlexnet2, self).__init__()
        # Possibly experiment - different initializations
        # TODO - regularization? see paper
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.cost = cost
        self.combine = combine
        self.sample = sample

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
        # NOTE - check - use bias ok?
        self.linear_map4 = Dense(units=384, use_bias=False)
        self.linear_map5 = Dense(units=256, use_bias=False)
        self.o4_to_out = Dense(units=4096, use_bias=False)
        self.o5_to_out = Dense(units=4096, use_bias=False)
        if(self.cost == 'pc'):
            if(self.sample == 'down'):
                self.u4 = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[1,1,384]))
                self.u5 = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[1,1,256]))
            elif(self.sample == 'up'):
                self.u4 = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[1,1,4096]))
                self.u5 = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[1,1,4096]))
        
        if(self.combine == 'concat'):
            self.attn_fc = Dense(units=num_classes, use_bias=False)
        elif(self.combine == 'indep'):
            self.attn_fc4 = Dense(units=num_classes, use_bias=False)
            self.attn_fc5 = Dense(units=num_classes, use_bias=False)


        self.drop7 = Dropout(rate=self.keep_prob)
        self.fc8 = Dense(units=self.num_classes)

        self.map_4_to_5 = Dense(units=256)
        self.add_4_5 = Add()
        
    def update_last_layer(self, num_classes):
        self.num_classes = num_classes
        self.fc8 = Dense(units=self.num_classes)
        if(self.combine == 'concat'):
            self.attn_fc = Dense(units=num_classes, use_bias=False)
        elif(self.combine == 'indep'):
            self.attn_fc4 = Dense(units=num_classes, use_bias=False)
            self.attn_fc5 = Dense(units=num_classes, use_bias=False)
        
    # Input - datum[0], datum[1] or datum[2], datum[3]
    def call(self, inputImg, mode='train'):  
        # print(tf.reduce_max(inputImg))
        # print(inputImg.shape)
        o1 = lrn(self.pool1(self.conv1(inputImg)), 2, 2e-05, 0.75)
        # print("o1", tf.shape(o1))
        o2 = lrn(self.pool2(self.conv2(self.pad(o1))), 2, 2e-05, 0.75)
        # print("o2", tf.shape(o2))
        o3 = self.conv3(o2)
        # print("o3", tf.shape(o3))
        # o3 = self.pool3(o3)
        # print("o3", tf.shape(o3))
        o4 = self.conv4(o3)
        # print("o4", tf.shape(o4))
        o5 = self.conv5(o4)
        # print("o5", tf.shape(o5))
        # Shifted pooling layers down
        # pooled_o5 = self.pool3(self.pool2(self.pool1(o5)))
        pooled_o5 = self.pool3(o5)
        # pooled_o5 = (o5)
        flat_o5 = self.flat(pooled_o5)
        # print("flat_o5", tf.shape(flat_o5))
        o6 = self.fc6(flat_o5)
        # print("o6", tf.shape(o6))
        # if(mode == 'train'):
        #     o6 = self.drop6(o6)
        o7 = self.fc7(o6)
        # print("o7", tf.shape(o7))
        # if(mode == 'train'):
        #     o7 = self.drop7(o7)
        
        # Use o4 and o5 to get attention

        # Desctiption: We scale o7 to the sizes of conv4 and conv5 separately, and 

        bsize = tf.shape(o4)[0]
        numc_4, numc_5 = tf.shape(o4)[3], tf.shape(o5)[3]
        o4_x, o4_y = tf.shape(o4)[1].numpy(), tf.shape(o4)[2].numpy()
        o5_x, o5_y = tf.shape(o5)[1].numpy(), tf.shape(o5)[2].numpy()
        
        # 1
        if(self.sample == 'down'):
            map_to_4 = self.linear_map4(o7) 
            map_to_5 = self.linear_map5(o7)
        else:
            # Alternate - map o4 and o5 up
            o4_upsample = self.o4_to_out(o4)
            o5_upsample = self.o5_to_out(o5)
        

        # 2
        if(self.sample == 'down'):
            map_to_4 = tf.expand_dims(map_to_4, axis=-1) # map_to_4 - [bsize, numc_4, 1]
            map_to_5 = tf.expand_dims(map_to_5, axis=-1) # map_to_5 - [bsize, numc_5, 1]
            # print("map_to_4", tf.shape(map_to_4))
            # print("map_to_5", tf.shape(map_to_5))
            o4 = tf.reshape(o4, [bsize, o4_x*o4_y, numc_4]) 
            o5 = tf.reshape(o5, [bsize, o5_x*o5_y, numc_5])
            # print("o4", tf.shape(o4))
            # print("o5", tf.shape(o5))

        else:
            # Alternate - sample up o4 and o5
            o4_upsample = tf.reshape(o4_upsample, [bsize, o4_x*o4_y, 4096])
            o5_upsample = tf.reshape(o5_upsample, [bsize, o5_x*o5_y, 4096])        

        # 3
        c_4 = [] 
        c_5 = []
        if(self.cost == 'dp'):
            if(self.sample == 'down'):
                c_4 = tf.matmul(o4, map_to_4)
                c_5 = tf.matmul(o5, map_to_5)
            else:
                # Alternate - upsample o4 and o5
                o7 = tf.reshape(o7, [bsize, 4096, 1])
                c_4 = tf.matmul(o4_upsample, o7)
                c_5 = tf.matmul(o5_upsample, o7)
        elif(self.cost == 'pc'):
            if(self.sample == 'down'):
                map_to_4 = tf.reshape(map_to_4, [bsize, 1, numc_4])
                map_to_5 = tf.reshape(map_to_5, [bsize, 1, numc_5])            
                c_4 = tf.reduce_sum(self.u4*(o4+map_to_4), axis=-1)
                c_5 = tf.reduce_sum(self.u5*(o5+map_to_5), axis=-1)
            else:
                # Alternative - upsample o4 and o5
                o7 = tf.reshape(o7, [bsize, 1, 4096])
                c_4 = tf.reduce_sum(self.u4*(o4_upsample+o7), axis=-1)
                c_5 = tf.reduce_sum(self.u5*(o5_upsample+o7), axis=-1)


        
        # 4
        c_4 = tf.reshape(c_4, [bsize, -1])
        c_5 = tf.reshape(c_5, [bsize, -1])
        # print("c_4", tf.shape(c_4))
        # print("c_5", tf.shape(c_5))
        a_4 = tf.nn.softmax(c_4)
        a_5 = tf.nn.softmax(c_5)        
        
        # 5
        # Shapes:
        # a_4 - [bsize, o4_x*o4_y, 1]
        # a_5 - [bsize, o5_x*o5_y, 1]
        a_4 = tf.expand_dims(a_4, axis=-1)
        a_5 = tf.expand_dims(a_5, axis=-1)
        # print("a_4", tf.shape(a_4))
        # print("a_5", tf.shape(a_5))
        o4 = tf.reshape(o4, [bsize, numc_4, o4_x*o4_y])
        o5 = tf.reshape(o5, [bsize, numc_5, o5_x*o5_y])
        # print("o4", tf.shape(o4))
        # print("o5", tf.shape(o5))

        # 6
        re_o4 = tf.matmul(o4, a_4)
        re_o5 = tf.matmul(o5, a_5)
        re_o4 = tf.squeeze(re_o4, [2])
        re_o5 = tf.squeeze(re_o5, [2])
        # print("re_o4", tf.shape(re_o4))
        # print("re_o5", tf.shape(re_o5))
        # re_o4 = self.map_4_to_5(re_o4)    # Project re_o4 to re_o5


        # penultimate = []
        
        if(self.combine == 'concat'):
            penultimate = tf.concat([re_o4, re_o5], axis=-1)
            # print("penultimate", tf.shape(penultimate))
            map_out = self.attn_fc(penultimate)
            # print("map_out", tf.shape(map_out))
            pred_class = (tf.argmax(map_out, axis=-1))
        elif(self.combine == 'indep'):
            map_out4 = self.attn_fc4(re_o4)
            map_out5 = self.attn_fc5(re_o5)
            prob_4, prob_5 = tf.nn.softmax(map_out4), tf.nn.softmax(map_out5)
            pred_class = (tf.argmax((prob_4+prob_5)/2.0, axis=-1))
        
        if(mode != 'maps'):
            if(self.combine == 'concat'):
                return map_out
            elif(self.combine == 'indep'):
                return (prob_4+prob_5)/2.0
        else:
            # Return attention maps
            # TODO - return SINGLE attention map!
            a_4 = tf.reshape(a_4, [bsize, o4_x, o4_y])
            a_5 = tf.reshape(a_5, [bsize, o5_x, o5_y])
            attention_map = tf.sqrt(a_4 * a_5)
            # c_4 = tf.reshape(c_4, [bsize, o4_x, o4_y])
            # c_5 = tf.reshape(c_5, [bsize, o5_x, o5_y])
            
            # attention_map = c_4*c_5
            # print(attention_map[0])
            # print(c_4[0])
            # print(c_5[0])
            
            xdim, ydim = o4_x, o4_y
            # c_4 = tf.reshape(c_4, [bsize, -1])
            # c_5 = tf.reshape(c_4, [bsize, -1])
            
            # min_val = tf.expand_dims(tf.reduce_min(c_4, axis=-1), axis=-1)
            # c_4 -= min_val
            # max_val = tf.expand_dims(tf.reduce_max(c_4, axis=-1), axis=-1)
            # c_4 /= max_val

            # min_val = tf.expand_dims(tf.reduce_min(c_5, axis=-1), axis=-1)
            # c_5 -= min_val
            # max_val = tf.expand_dims(tf.reduce_max(c_5, axis=-1), axis=-1)
            # c_5 /= max_val

            # attention_map = c_4

            attention_map = tf.reshape(attention_map, [bsize, -1])
            min_val = tf.expand_dims(tf.reduce_min(attention_map, axis=-1), axis=-1)
            attention_map -= min_val
            max_val = tf.expand_dims(tf.reduce_max(attention_map, axis=-1), axis=-1)
            attention_map /= max_val
            attention_map = tf.reshape(attention_map, [bsize, xdim, ydim])            
            return attention_map, pred_class
            

def loss_attnalex2(alexCNN, datum, mode):
    if(alexCNN.combine == 'concat'):
        # Assuming datum[0] is data. datum[1] is labels
        # logits1, logits2 = alexCNN(datum[0], mode)
        # logits = tf.concat([logits1, logits2], axis=-1)
        # logits = logits1+logits2
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=datum[1])
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=alexCNN(datum[0], mode), labels=datum[1])
        # return tf.reduce_sum(loss)/ tf.cast(tf.size(datum[1]), dtype=tf.float32)

        logits = alexCNN(datum[0], mode)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=datum[1])
        # print(tf.shape(datum[1]))
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=alexCNN(datum[0], mode), labels=datum[1])
        return tf.reduce_sum(loss)/ tf.cast(tf.size(datum[1]), dtype=tf.float32)
    elif(alexCNN.combine == 'indep'):
        probs = alexCNN(datum[0], mode)
        row_indices = tf.range(tf.size(datum[1]))
        full_indices = tf.stack([row_indices, datum[1]], axis=1)
        # print(probs)
        # print(full_indices)
        loss = tf.gather_nd(probs, full_indices) # Probability of the CORRECT label!
        # print(loss)
        return (-1.0)*tf.reduce_sum(tf.log(loss))/ tf.cast(tf.size(datum[1]), dtype=tf.float32)

attnalex2_loss_grads = tfe.implicit_value_and_gradients(loss_attnalex2)