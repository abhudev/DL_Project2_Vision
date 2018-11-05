from PIL import Image
import pickle
import argparse
import numpy as np
from skimage import filters
from skimage import transform
import os

parser = argparse.ArgumentParser()
parser.add_argument('--attn_map', type=str)
parser.add_argument('--category', type=str)
parser.add_argument('--thresh', type=float, default=0.5)
args = parser.parse_args()


if(args.attn_map is None):
    print("Provide --attn_map")
    exit()

cats = ['airplane', 'car', 'horse']

ground_truth = {
                    'airplane': 'train_text/Airplane_ground.txt',
                    'car': 'train_text/Car_ground.txt',
                    'horse': 'train_text/Horse_ground.txt'
               }

if(args.category not in cats):
    print(f"Provide --category in {cats}")
    exit()

with open(ground_truth[args.category], 'r') as fg:
    lines = fg.readlines()
    num_imgs = len(lines)
    ground_imgs = []
    for i, line in enumerate(lines):
        with Image.open(line.strip('\n')) as im:
            ground_imgs.append(np.asarray(im))
            # print(ground_imgs[i].shape)
            # if(i == 10):
            #     exit()

maps, names = [], []

with open(args.attn_map, "rb") as fp:      
    while(True):
        try:
            gen_maps = pickle.load(fp)
            batch_names = pickle.load(fp)
            for i in range(gen_maps.shape[0]):
                maps.append(gen_maps[i, :])
                names.append(batch_names[i])
            # print(maps[i].shape)
            gen_maps = None
        except EOFError:
            # Print all the details of accuracy etc            
            print(len(ground_imgs), len(maps))
            if(len(ground_imgs) != len(maps)):
                print("Error - number of ground and original images do not match")
                exit()
            num_imgs = len(ground_imgs)
            for i in range(num_imgs):
                n1 = os.path.basename(os.path.normpath(names[i].decode('UTF-8')))[:-4]
                n2 = os.path.basename(os.path.normpath(lines[i].strip('\n')))[:-4]
                if(n1 != n2):
                    print("Ground truth and seg map do not match")
                    print(n1, n2, i)                    
                    exit()
            print("Names match!")
            tot_c = 0
            correct_c = 0
            for i in range(num_imgs):
                prediction, gt = maps[i], ground_imgs[i]
                # print(prediction.shape)
                # print(gt.shape)
                # exit()
                prediction = transform.resize(prediction, gt.shape)
                threshold = filters.threshold_otsu(prediction)
                prediction[prediction < threshold] = 0
                prediction[prediction >= threshold] = 1
                # IOU
                intersection = np.sum(prediction * gt)
                union = np.sum(prediction) + np.sum(gt) - intersection
                iou = intersection/union
                if(iou >= 0.5):
                    correct_c += 1
                tot_c += 1
            print(f"Score = {correct_c/tot_c}")
            exit()
            

