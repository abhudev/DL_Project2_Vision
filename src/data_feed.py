import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import pickle


#--------------------Preprocessing functions--------------------#
def parse_cifar10(filename, label):
    image_string = tf.read_file(filename)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_png(image_string, channels=3)
    # print(image)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    # print(image)
    # exit()
    image = tf.image.resize_images(image, [227, 227])
    return image, label


def parse_cub(filename, label):
    image_string = tf.read_file(filename)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [227, 227])
    return image, label-1


def parse_svhn(filename, label):
    image_string = tf.read_file(filename)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [227, 227])
    label = label % 10
    return image, label


def preprocess_cifar100(img, label):
    # print(img)
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # img /= 255.0
    # print(img)
    # exit()
    img = tf.image.resize_images(img, [227, 227])
    img = tf.divide(img, 255.0)
    return img, label


def parse_odd(in_img, ground):
    string1, string2 = tf.read_file(in_img), tf.read_file(ground)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    input_img = tf.image.decode_jpeg(string1, channels=3)
    ground_img = tf.image.decode_png(string2, channels=1)
    # This will convert to float values in [0, 1]
    input_img = tf.image.convert_image_dtype(input_img, tf.float32)
    # ground_img = tf.image.convert_image_dtype(ground_img, tf.float32)
    input_img = tf.image.resize_images(input_img, [227, 227])
    ground_img = tf.image.resize_images(ground_img, [227, 227])
    return input_img, ground_img
#---------------------------------------------------------------#

func = {'cifar10': parse_cifar10, 'cub': parse_cub, 'svhn': parse_svhn}

def get_img_data(image_file, class_file, batch_size, in_data, num_threads=4, mode='train'):
    with open(image_file, 'r') as fim, open(class_file, 'r') as fcin:
        filenames, labels = [], []
        for line in fim:
            filenames.append(line.strip('\n'))
        for line in fcin:
            labels.append(int(line.strip('\n')))
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(len(filenames))
    parse_func = func[in_data]
    dataset = dataset.map(parse_func, num_parallel_calls=num_threads)
    # dataset = dataset.map(train_preprocess, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def get_cifar100_data(data_file, batch_size, num_threads=4, mode='train'):
    with open(data_file, "rb") as fp:
        # print(data_file)
        fp.seek(0)
        data = pickle.load(fp, encoding='bytes')
        # Optional - do only for first 20 instances
        data[b'data'] = data[b'data'][:20]
        data[b'fine_labels'] = data[b'fine_labels'][:20]
        # print(data[b'fine_labels'])
        # exit()
    dataset = tf.data.Dataset.from_tensor_slices((data[b'data'], data[b'fine_labels']))
    dataset = dataset.map(lambda im, lb: (tf.reshape(im, [32,32,3]), lb), num_parallel_calls=num_threads)
    # print(dataset[0][0])
    dataset = dataset.shuffle( len(data[b'data']) )
    dataset = dataset.map(preprocess_cifar100, num_parallel_calls=num_threads)
    # print(dataset[0][0])
    # exit()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

# This must be done for each category! Separately! Do the evaluation for 
# Airplane, Car, Horse
def get_obj_data(inp_img, ground_truth, batch_size, num_threads=4, mode='train'):
    with open(inp_img, 'r') as fim, open(ground_truth, 'r') as fg:
        timg, gimg = [], []
        for line in fim:
            timg.append(line.strip('\n'))
        for line in fg:
            gimg.append(line.strip('\n'))
    
    dataset = tf.data.Dataset.from_tensor_slices((timg, gimg))
    dataset = dataset.shuffle(len(timg))
    dataset = dataset.map(parse_odd, num_parallel_calls=num_threads)
    # dataset = dataset.map(train_preprocess, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


#--------------------OLD CODE--------------------

# def get_cifar_10_data(image_file, class_file, batch_size, num_threads=4, mode='train'):
#     with open(image_file, 'r') as fim, open(class_file, 'r') as fcin:
#         filenames, labels = [], []
#         for line in fim:
#             filenames.append(line.strip('\n'))
#         for line in fcin:
#             labels.append(int(line.strip('\n')))
    
#     dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
#     dataset = dataset.shuffle(len(filenames))
#     dataset = dataset.map(parse_cifar10, num_parallel_calls=num_threads)
#     # dataset = dataset.map(train_preprocess, num_parallel_calls=4)
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.prefetch(1)
#     return dataset

# Get CUB dataset
# def get_cub_data(image_file, class_file, box_file, batch_size, num_threads=4, mode='train'):
#     with open(image_file, 'r') as fim, open(class_file, 'r') as fcin, open(box_file, 'r') as fbox:
#         filenames, labels, box = [], [], []
#         for line in fim:
#             filenames.append(line.strip('\n'))
#         for line in fcin:
#             labels.append(line.strip('\n'))
#         for line in fbox:
#             box.append([float(val) for val in line.split()])    
    
#     dataset = tf.data.Dataset.from_tensor_slices((filenames, labels, box))
#     dataset = dataset.shuffle(len(filenames))
#     dataset = dataset.map(parse_cub, num_parallel_calls=num_threads)
#     # dataset = dataset.map(train_preprocess, num_parallel_calls=4)
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.prefetch(1)
#     return dataset
