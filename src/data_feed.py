import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import h5py

def parse_function(filename, label):
    image_string = tf.read_file(filename)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_png(image_string, channels=3)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [64, 64])
    return image, label

def get_cifar_10_data(image_file, class_file, batch_size, num_threads=4, mode='train'):
    with open(image_file, 'r') as fim, open(class_file, 'r') as fcin:
        filenames, labels = [], []
        for line in fim:
            filenames.append(line.strip('\n'))
        for line in fcin:
            labels.append(line.strip('\n'))
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(parse_function, num_parallel_calls=num_threads)
    # dataset = dataset.map(train_preprocess, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

# Preprocess
def preprocess_cifar100(img, label):
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, label

def get_cifar_100_data(data_file, batch_size, num_threads=4, mode='train'):
    with open(data_file, "rb") as fp:
        fp.seek(0)
        data = pickle.load(fp, encoding='bytes')
    
    dataset = tf.data.Dataset.from_tensor_slices((data[b'data'], data[b'fine_labels']))
    dataset = dataset.map(lambda im, lb: (np.reshape(im, [32,32,3]), lb), num_parallel_calls=num_threads)
    dataset = dataset.shuffle( data[b'data'].shape[0] )
    dataset = dataset.map(preprocess_cifar100, num_parallel_calls=num_threads)
    return dataset

def parse_function(filename, label, box):
    image_string = tf.read_file(filename)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [64, 64])
    return image, label


# Get CUB dataset
def get_cub_data(image_file, class_file, box_file, batch_size, num_threads=4, mode='train'):
    with open(image_file, 'r') as fim, open(class_file, 'r') as fcin, open(box_file, 'r') as fbox:
        filenames, labels, box = [], [], []
        for line in fim:
            filenames.append(line.strip('\n'))
        for line in fcin:
            labels.append(line.strip('\n'))
        for line in fbox:
            box.append([float(val) for val in line.split()])    
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels, box))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(parse_function_cub, num_parallel_calls=num_threads)
    # dataset = dataset.map(train_preprocess, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset

# This must be done for each category! Separately! Do the evaluation for 
# Airplane, Car, Horse
def get_obj_data():
    

def get_svhn_data(image_fol, mat):
    cwd = os.getcwd()
    fp  = h5py.File(mat)

