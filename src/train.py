import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
tf.set_random_seed(49)


import numpy as np
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--task', type=str)
parser.add_argument('--')
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument()
args = parser.parse_args()



opt = []
if(args.opt == 'adam'):
    opt = tf.train.AdamOptimizer(learning_rate=args.lr)
elif(args.opt == 'sgd'):
    opt = tf.train.GradientDescentOptimizer(learning_rate=args.lr)

