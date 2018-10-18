import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
tf.set_random_seed(49)
import data_feed
import model
import numpy as np
import os
import time
import argparse

parser = argparse.ArgumentParser()

# :: Data used ::
parser.add_argument('--data', type=str)

# :: CIFAR 10 data ::
parser.add_argument('--cifar10_train_img', type=str, default='train_text/train_img_cifar10.txt')
parser.add_argument('--cifar10_train_classes', type=str, default='train_text/train_classes_cifar10.txt')
parser.add_argument('--cifar10_dev_img', type=str, default='train_text/dev_img_cifar10.txt')
parser.add_argument('--cifar10_dev_classes', type=str, default='train_text/dev_classes_cifar10.txt')
parser.add_argument('--cifar10_test_img', type=str, default='train_text/test_img_cifar10.txt')
parser.add_argument('--cifar10_test_classes', type=str, default='train_text/test_classes_cifar10.txt')

# :: CIFAR 100 data ::
parser.add_argument('--cifar100_train', type=str, default="../../Proj2_data/Datasets/CIFAR/cifar-100-python/train")
parser.add_argument('--cifar100_test', type=str, default="../../Proj2_data/Datasets/CIFAR/cifar-100-python/test")
parser.add_argument('--cifar100_meta', type=str, default="../../Proj2_data/Datasets/CIFAR/cifar-100-python/meta")

# :: CUB data ::
parser.add_argument('--cub_train_img', type=str, default='train_text/train_img_CUB.txt')
parser.add_argument('--cub_train_classes', type=str, default='train_text/train_classes_CUB.txt')
parser.add_argument('--cub_train_bbox', type=str, default='train_text/train_bbox_CUB.txt')

parser.add_argument('--cub_test_img', type=str, default='train_text/test_img_CUB.txt')
parser.add_argument('--cub_test_classes', type=str, default='train_text/test_classes_CUB.txt')
parser.add_argument('--cub_test_bbox', type=str, default='train_text/test_bbox_CUB.txt')

# :: ODD data ::
parser.add_argument('--odd_airplane_img', type=str, default='train_text/Airplane_train.txt')
parser.add_argument('--odd_airplane_ground', type=str, default='train_text/Airplane_ground.txt')
parser.add_argument('--odd_horse_img', type=str, default='train_text/Horse_train.txt')
parser.add_argument('--odd_horse_ground', type=str, default='train_text/Horse_ground.txt')
parser.add_argument('--odd_car_img', type=str, default='train_text/Car_train.txt')
parser.add_argument('--odd_car_ground', type=str, default='train_text/Car_ground.txt')

# :: Checkpoints ::
parser.add_argument('--ckpt_dir_cifar10', type=str, default='../Checkpoints_cifar10')
parser.add_argument('--ckpt_dir_cifar10_regular', type=str, default='../Checkpoints_cifar10_regular')
parser.add_argument('--ckpt_dir_cifar100', type=str, default='../Checkpoints_cifar100')
parser.add_argument('--ckpt_dir_cifar100_regular', type=str, default='../Checkpoints_cifar100_regular')
parser.add_argument('--ckpt_dir_cub', type=str, default='../Checkpoints_cub')
parser.add_argument('--ckpt_dir_cub_regular', type=str, default='../Checkpoints_cub_regular')
parser.add_argument('--ckpt_dir_svhn', type=str, default='../Checkpoints_svhn')
parser.add_argument('--ckpt_dir_svhn_regular', type=str, default='../Checkpoints_svhn_regular')
parser.add_argument('--ckpt_file', type=str, default='ckpt')


# :: Train parameters ::
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--drop', type=float, default=0.5)
parser.add_argument('--bsize', type=int, default=64)
parser.add_argument('--device', type=str, default="/gpu:0")
parser.add_argument('--num_epochs', type=str, default=200)
args = parser.parse_args()

if(args.data != 'cifar10'):
    print("Initial stages; please provide cifar10 data only!")
    exit()

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

num_classes = class_no[args.data]
ckpt_prefix = checkpoint_dir[args.data]+args.ckpt_file
regular_ckpt_prefix = regular_checkpoint_dir[args.data]+args.ckpt_file
vanilla_alex = model.BaseAlexnet(num_classes, args.drop)

opt = []
if(args.opt == 'adam'):
    opt = tf.train.AdamOptimizer(learning_rate=args.lr)
elif(args.opt == 'sgd'):
    opt = tf.train.GradientDescentOptimizer(learning_rate=args.lr)

train_data = data_feed.get_cifar_10_data(args.cifar10_train_img, args.cifar10_train_classes, args.bsize)
dev_data = data_feed.get_cifar_10_data(args.cifar10_dev_img, args.cifar10_dev_classes, args.bsize, mode='eval')

saver = tfe.Checkpoint(optimizer=opt, model=vanilla_alex, optimizer_step=tf.train.get_or_create_global_step())
saver.restore(tf.train.latest_checkpoint(regular_ckpt_prefix))

STATS_STEPS = 1
EVAL_STEPS = 50


with tf.device(args.device):
    start_reg = time.time()
    for epoch_num in range(args.num_epochs):
        # batch_loss = []        
        # if(epoch_num > 0):
        #     saver.restore(tf.train.latest_checkpoint(ckpt_prefix))            
            
        log_msg(f"Begin Epoch {epoch_num} with restored model")
        start_reg = time.time()
        # :: CIFAR 10 epoch ::
        for step_num, datum in enumerate(train_data, start=1):            
            loss_value, gradients = model.alex_loss_grads(vanilla_alex, datum, 'train')
            opt.apply_gradients(gradients, global_step=tf.train.get_or_create_global_step())    

            if step_num % STATS_STEPS == 0:
                acc = tfe.metrics.Accuracy()
                for dev_d in dev_data:
                    logits = vanilla_alex(dev_d[0], 'eval')
                    preds = tf.argmax(logits, axis=1)
                    acc(tf.reshape(tf.cast(dev_d[1], dtype=tf.int64), [-1,]), preds)
                log_msg(f'Epoch: {epoch_num} Step: {step_num} Avg Loss: {np.average(np.asarray(loss_value))} Dev accurac: {acc.result().numpy()}')
                batch_loss = []
        
            # if step_num % EVAL_STEPS == 0:
            #     # Compute test accuracy
            #     #Save model!
            #     if ppl < valid_ppl_nmt:
            #         saver.save(ckpt_prefix)                
            #         log_msg(f'Epoch: {epoch_num} Step: {step_num} ppl improved: {} old: {} Model saved')
            #     else:
            #         log_msg(f'Epoch: {epoch_num} Step: {step_num} ppl worse: {} old: {}')
                
            if((time.time() - start_reg)/3600 >= 1.0):
                saver.save(reg_ckpt)                
                log_msg(f'Epoch: {epoch_num} Step: {step_num} Model regularly saved')
                start_reg = time.time()