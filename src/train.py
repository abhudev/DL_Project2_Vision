import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
tf.set_random_seed(49)
import data_feed
import model
import numpy as np
import os
import sys
import time
import argparse
import pickle
# from skimage import filters
# import skimage
import numpy as np

parser = argparse.ArgumentParser()

# :: Data used ::
parser.add_argument('--data', type=str)

# :: Curves ::
parser.add_argument('--train_curve', type=str, default='train_curve.csv')
parser.add_argument('--dev_curve', type=str, default ='dev_curve.csv')
parser.add_argument('--test_curve', type=str, default='test_curve.csv')
parser.add_argument('--dev_acc', type=str, default ='dev_acc.csv')
parser.add_argument('--test_acc', type=str, default='test_acc.csv')


# :: CIFAR 10 data ::
parser.add_argument('--cifar10_train_img', type=str, default='train_text/train_img_cifar10.txt')
parser.add_argument('--cifar10_train_classes', type=str, default='train_text/train_classes_cifar10.txt')
parser.add_argument('--cifar10_dev_img', type=str, default='train_text/dev_img_cifar10.txt')
parser.add_argument('--cifar10_dev_classes', type=str, default='train_text/dev_classes_cifar10.txt')
parser.add_argument('--cifar10_test_img', type=str, default='train_text/test_img_cifar10.txt')
parser.add_argument('--cifar10_test_classes', type=str, default='train_text/test_classes_cifar10.txt')

# :: CIFAR 100 data ::
parser.add_argument('--cifar100_train', type=str, default="../../Proj2_data/Datasets/CIFAR/cifar-100-python/train_new")
parser.add_argument('--cifar100_dev', type=str, default="../../Proj2_data/Datasets/CIFAR/cifar-100-python/dev_new")
parser.add_argument('--cifar100_test', type=str, default="../../Proj2_data/Datasets/CIFAR/cifar-100-python/test")
parser.add_argument('--cifar100_meta', type=str, default="../../Proj2_data/Datasets/CIFAR/cifar-100-python/meta/")

# :: CUB data ::
parser.add_argument('--cub_train_img', type=str, default='train_text/train_crop_img_CUB.txt')
parser.add_argument('--cub_train_classes', type=str, default='train_text/train_crop_classes_CUB.txt')
# parser.add_argument('--cub_train_bbox', type=str, default='train_text/train_bbox_CUB.txt')

parser.add_argument('--cub_dev_img', type=str, default='train_text/dev_crop_img_CUB.txt')
parser.add_argument('--cub_dev_classes', type=str, default='train_text/dev_crop_classes_CUB.txt')
# parser.add_argument('--cub_dev_bbox', type=str, default='train_text/dev_bbox_CUB.txt')

parser.add_argument('--cub_test_img', type=str, default='train_text/test_crop_img_CUB.txt')
parser.add_argument('--cub_test_classes', type=str, default='train_text/test_crop_classes_CUB.txt')
# parser.add_argument('--cub_test_bbox', type=str, default='train_text/test_bbox_CUB.txt')

# :: SVHN data ::
parser.add_argument('--svhn_train_img', type=str, default='train_text/SVHN_train_img.txt')
parser.add_argument('--svhn_train_classes', type=str, default='train_text/SVHN_train_classes.txt')
parser.add_argument('--svhn_dev_img', type=str, default='train_text/SVHN_dev_img.txt')
parser.add_argument('--svhn_dev_classes', type=str, default='train_text/SVHN_dev_classes.txt')
parser.add_argument('--svhn_test_img', type=str, default='train_text/SVHN_test_img.txt')
parser.add_argument('--svhn_test_classes', type=str, default='train_text/SVHN_test_classes.txt')

# :: ODD data ::
parser.add_argument('--odd_category', type=str)
parser.add_argument('--odd_airplane_img', type=str, default='train_text/Airplane_train.txt')
parser.add_argument('--odd_airplane_ground', type=str, default='train_text/Airplane_ground.txt')
parser.add_argument('--odd_horse_img', type=str, default='train_text/Horse_train.txt')
parser.add_argument('--odd_horse_ground', type=str, default='train_text/Horse_ground.txt')
parser.add_argument('--odd_car_img', type=str, default='train_text/Car_train.txt')
parser.add_argument('--odd_car_ground', type=str, default='train_text/Car_ground.txt')

# :: Checkpoints ::
parser.add_argument('--ckpt_dir_cifar10', type=str, default='../CHECKPOINTS/Checkpoints_cifar10/')
parser.add_argument('--ckpt_dir_cifar10_regular', type=str, default='../CHECKPOINTS/Checkpoints_cifar10_regular/')
parser.add_argument('--ckpt_dir_cifar100', type=str, default='../CHECKPOINTS/Checkpoints_cifar100/')
parser.add_argument('--ckpt_dir_cifar100_regular', type=str, default='../CHECKPOINTS/Checkpoints_cifar100_regular/')
parser.add_argument('--ckpt_dir_cub', type=str, default='../CHECKPOINTS/Checkpoints_cub/')
parser.add_argument('--ckpt_dir_cub_regular', type=str, default='../CHECKPOINTS/Checkpoints_cub_regular/')
parser.add_argument('--ckpt_dir_svhn', type=str, default='../CHECKPOINTS/Checkpoints_svhn/')
parser.add_argument('--ckpt_dir_svhn_regular', type=str, default='../CHECKPOINTS/Checkpoints_svhn_regular/')
parser.add_argument('--ckpt_file', type=str, default='ckpt')


# :: Train parameters ::
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--drop', type=float, default=0.5)
parser.add_argument('--bsize', type=int, default=64)
parser.add_argument('--device', type=str, default="/gpu:0")
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--model', type=str, default='attn')
parser.add_argument('--attn_combine', type=str, default='concat')
parser.add_argument('--attn_sample', type=str, default='down')
parser.add_argument('--attn_cost', type=str, default='dp')
parser.add_argument('--stats', type=int, default=25)
parser.add_argument('--eval', type=int, default=100)
parser.add_argument('--attnmap_output', type=str)
args = parser.parse_args()

# if(args.data != 'cifar10'):
#     print("Initial stages; please provide cifar10 data only!")
#     exit()

# :: TensorFlow logging utility ::
logging = tf.logging
logging.set_verbosity(logging.INFO)
def log_msg(msg):
    logging.info(f'{time.ctime()}: {msg}')

class_no = {
                'cifar10': 10,
                'cifar100': 100,
                'cub': 200,
                'svhn': 10,
                'odd': None                
           }

checkpoint_dir = {
                        'cifar10': args.ckpt_dir_cifar10,
                        'cifar100': args.ckpt_dir_cifar100,
                        'cub': args.ckpt_dir_cub,
                        'svhn': args.ckpt_dir_svhn                                                
                 }

regular_checkpoint_dir = {
                        'cifar10': args.ckpt_dir_cifar10_regular,
                        'cifar100': args.ckpt_dir_cifar100_regular,
                        'cub': args.ckpt_dir_cub_regular,
                        'svhn': args.ckpt_dir_svhn_regular
                 }

train_data = {
                'cifar10': (args.cifar10_train_img, args.cifar10_train_classes, args.cifar10_dev_img, args.cifar10_dev_classes, args.cifar10_test_img, args.cifar10_test_classes),
                'cifar100': (args.cifar100_train, args.cifar100_dev, args.cifar100_test),
                'cub': (args.cub_train_img, args.cub_train_classes, args.cub_dev_img, args.cub_dev_classes, args.cub_test_img, args.cub_test_classes),
                'svhn': (args.svhn_train_img, args.svhn_train_classes, args.svhn_dev_img, args.svhn_dev_classes, args.svhn_test_img, args.svhn_test_classes),
             }

#------------------------------OBJECT DETECTION DATASET------------------------------#

odd_data_dict = {
                    'airplane': (args.odd_airplane_img, args.odd_airplane_ground),
                    'horse': (args.odd_horse_img, args.odd_horse_ground),
                    'car': (args.odd_car_img, args.odd_car_ground)
                }

if(args.data == 'odd'):
    
    if(args.attnmap_output is None):
        print("Provide output for attention maps")
        exit()

    category = args.odd_category
    odd_cats = list(odd_data_dict.keys())
    if(category not in odd_cats):
        print("Please specify a valid ODD category")
        exit()
    # NOTE - Be sure to get the parameters right for attention!
    # Number of classes don't matter
    # The model is to be loaded from CIFAR trained models
    alex_model = []
    if(args.model == 'vanilla'):
        alex_model = model.BaseAlexnet(3, args.drop)    
    elif (args.model == 'attn'):
        alex_model = model.AttnAlexnet(3, args.drop, combine=args.attn_combine, sample=args.attn_sample)    
    elif (args.model == 'gap'):
        alex_model = model.GAPAlexnet(3, args.drop)    
    
    eval_img, ground_truth = odd_data_dict[category][0], odd_data_dict[category][1]    
    eval_data = data_feed.get_obj_data(eval_img, ground_truth, args.bsize)    
    fmap = open(args.attnmap_output, 'wb')

    with tf.device(args.device):        
        for i, datum in enumerate(eval_data):
            if((i+1)%100 == 0):
                sys.stdout.write(f'\rBatch {i+1}')
                sys.stdout.flush()
            attention_map = alex_model(datum[0], mode='maps')
            # thresh_list = []
            attention_map = attention_map.numpy()
            gt_map = datum[1].numpy()
            # Dump to file
            # Dump attention map and 
            pickle.dump(attention_map, fmap)
            pickle.dump(gt_map, fmap)
            # for i in range(bsize):
            #     thresh_list.append(skimage.filters.threshold_otsu(attention_map[i]))
            # thresh_list = np.expand_dims(np.expand_dims(np.array(thresh_list), axis=-1), axis=-1)
            # attention_map[attention_map < thresh_list] = 0
            # attention_map[attention_map >= thresh_list] = 1
            # attention_map = tf.convert_to_tensor(attention_map)
            # TODO - IOU scores
    print('')
    fmap.close()            
    exit() 




#------------------------------OBJECT DETECTION DATASET------------------------------#

num_classes = class_no[args.data]
print(f"Number of classes = {num_classes}")
try:
    os.makedirs(checkpoint_dir[args.data])
except FileExistsError:
    pass
try:
    os.makedirs(regular_checkpoint_dir[args.data])
except FileExistsError:
    pass

ckpt_prefix = os.path.join(checkpoint_dir[args.data], args.ckpt_file)
regular_ckpt_prefix = os.path.join(regular_checkpoint_dir[args.data], args.ckpt_file)
model_data = train_data[args.data]

alex_model = []
loss_fn = []
if(args.model == 'vanilla'):
    alex_model = model.BaseAlexnet(num_classes, args.drop)
    loss_fn = model.alex_loss_grads
elif (args.model == 'attn'):
    alex_model = model.AttnAlexnet(num_classes, args.drop, combine=args.attn_combine, sample=args.attn_sample)
    loss_fn = model.attnalex_loss_grads
elif (args.model == 'gap'):
    alex_model = model.GAPAlexnet(num_classes, args.drop)
    loss_fn = model.gapalex_loss_grads

opt = []
if(args.opt == 'adam'):
    opt = tf.train.AdamOptimizer(learning_rate=args.lr)
elif(args.opt == 'sgd'):
    opt = tf.train.GradientDescentOptimizer(learning_rate=args.lr)
train_data, dev_data, test_data = [], [], []
if(args.data in ['cifar10', 'cub', 'svhn']):
    train_data = data_feed.get_img_data(model_data[0], model_data[1], args.bsize, args.data)
    dev_data = data_feed.get_img_data(model_data[2], model_data[3], args.bsize, args.data, mode='eval')
    test_data = data_feed.get_img_data(model_data[4], model_data[5], args.bsize, args.data, mode='eval')
elif(args.data == 'cifar100'):
    train_data = data_feed.get_cifar100_data(args.cifar10_train_img, args.cifar10_train_classes, args.bsize, args.data)
    dev_data = data_feed.get_cifar100_data(args.cifar10_dev_img, args.cifar10_dev_classes, args.bsize, args.data, mode='eval')

saver = tfe.Checkpoint(optimizer=opt, model=alex_model, optimizer_step=tf.train.get_or_create_global_step())
# saver.restore(tf.train.latest_checkpoint(regular_ckpt_prefix))
# saver.restore(tf.train.latest_checkpoint(ckpt_prefix))

STATS_STEPS = args.stats
EVAL_STEPS = args.eval

init_acc = 0
fp_train = open(args.train_curve, 'w')
fp_dev = open(args.dev_curve, 'w')
fp_test = open(args.test_curve, 'w')
fp_dev_acc = open(args.dev_acc, 'w')
fp_test_acc = open(args.test_acc, 'w')


with tf.device(args.device):
    start_reg = time.time()
    for epoch_num in range(args.num_epochs):
        # batch_loss = []        
        # if(epoch_num > 0):
        #     saver.restore(tf.train.latest_checkpoint(ckpt_prefix))            
            
        log_msg(f"Begin Epoch {epoch_num}")
        start_reg = time.time()
        # :: CIFAR 10 epoch ::
        for step_num, datum in enumerate(train_data, start=1):            
            # loss_value, gradients = model.alex_loss_grads(alex_model, datum, 'train')
            # loss_value, gradients = model.attnalex_loss_grads(alex_model, datum, 'train')
            loss_value, gradients = loss_fn(alex_model, datum, 'train')
            opt.apply_gradients(gradients, global_step=tf.train.get_or_create_global_step())    

            if step_num % STATS_STEPS == 0:
                loss_avg = np.average(np.asarray(loss_value))
                log_msg(f'Epoch: {epoch_num} Step: {step_num} Avg Loss: {loss_avg}')
                fp_train.write(str(loss_avg)+'\n')
                batch_loss = []
        
            if step_num % EVAL_STEPS == 0:
                # Compute test accuracy
                #Save model!
                acc = tfe.metrics.Accuracy()
                dev_loss = 0
                num_data = 0
                for dev_d in dev_data:
                    logits = alex_model(dev_d[0], 'eval')
                    preds = tf.argmax(logits, axis=1)
                    acc(tf.reshape(tf.cast(dev_d[1], dtype=tf.int64), [-1,]), preds)
                    dloss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=dev_d[1])
                    dev_loss += tf.reduce_sum(dloss)
                    num_data += tf.cast(tf.size(dev_d[1]), dtype=tf.float32)
                new_acc = acc.result().numpy()
                fp_dev_acc.write(str(new_acc)+'\n')
                fp_dev.write(str(dev_loss/num_data)+'\n')
                if new_acc > init_acc:
                    saver.save(ckpt_prefix)                
                    log_msg(f'Epoch: {epoch_num} Step: {step_num} acc improved: {new_acc} old: {init_acc} Model saved')
                    init_acc = new_acc
                else:
                    log_msg(f'Epoch: {epoch_num} Step: {step_num} acc worse: {new_acc} old: {init_acc}')                    

                acc_test = tfe.metrics.Accuracy()
                test_loss = 0
                num_data = 0
                for test_d in test_data:
                    logits = alex_model(test_d[0], 'eval')
                    preds = tf.argmax(logits, axis=1)
                    acc_test(tf.reshape(tf.cast(test_d[1], dtype=tf.int64), [-1,]), preds)
                    dloss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=test_d[1])
                    test_loss += tf.reduce_sum(dloss)
                    num_data += tf.cast(tf.size(test_d[1]), dtype=tf.float32)
                new_acc_test = acc_test.result().numpy()
                fp_test_acc.write(str(new_acc_test)+'\n')
                fp_test.write(str(test_loss/num_data)+'\n')
                log_msg(f'Epoch: {epoch_num} Step: {step_num} test acc: {new_acc_test}')        
                
            if((time.time() - start_reg)/3600 >= 1.0):
                saver.save(regular_ckpt_prefix)                
                log_msg(f'Epoch: {epoch_num} Step: {step_num} Model regularly saved')
                start_reg = time.time()