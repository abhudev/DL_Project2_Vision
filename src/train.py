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

# :: CIFAR 10 data ::
parser.add_argument('--cifar10_train_img', type=str, default='cifar10_train_img.txt')
parser.add_argument('--cifar10_train_class', type=str, default='cifar10_train_classes.txt')
parser.add_argument('--cifar10_test_img', type=str, default='cifar10_test_img.txt')
parser.add_argument('--cifar10_test_class', type=str, default='cifar10_test_classes.txt')
parser.add_argument('--cifar100_train', type=str, default="../../Proj2_data/Datasets/CIFAR/cifar-100-python/train")

# :: CIFAR 100 data ::
parser.add_argument('--cifar100_test', type=str, default="../../Proj2_data/Datasets/CIFAR/cifar-100-python/test")
parser.add_argument('--cifar100_meta', type=str, default="../../Proj2_data/Datasets/CIFAR/cifar-100-python/meta")

# :: CUB data ::
parser.add_argument('--cub_train_img', type=str, default='CUB_train_img.txt')
parser.add_argument('--cub_train_classes', type=str, default='CUB_train_classes.txt')
parser.add_argument('--cub_train_bbox', type=str, default='CUB_train_bbox.txt')

parser.add_argument('--cub_test_img', type=str, default='CUB_test_img.txt')
parser.add_argument('--cub_test_classes', type=str, default='CUB_test_classes.txt')
parser.add_argument('--cub_test_bbox', type=str, default='CUB_test_bbox.txt')

# :: ODD data ::
parser.add_argument('--odd_airplane_img', type=str, defualt='Airplane_train.txt')
parser.add_argument('--odd_airplane_ground', type=str, defualt='Airplane_ground.txt')
parser.add_argument('--odd_horse_img', type=str, defualt='Horse_train.txt')
parser.add_argument('--odd_horse_ground', type=str, defualt='Horse_ground.txt')
parser.add_argument('--odd_car_img', type=str, defualt='Car_train.txt')
parser.add_argument('--odd_car_ground', type=str, defualt='Car_ground.txt')


parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument()
args = parser.parse_args()

# :: TensorFlow logging utility ::
logging = tf.logging
logging.set_verbosity(logging.INFO)
def log_msg(msg):
    logging.info(f'{time.ctime()}: {msg}')


opt = []
if(args.opt == 'adam'):
    opt = tf.train.AdamOptimizer(learning_rate=args.lr)
elif(args.opt == 'sgd'):
    opt = tf.train.GradientDescentOptimizer(learning_rate=args.lr)

