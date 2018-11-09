from PIL import Image
import pickle
import argparse
import numpy as np
from skimage import filters
from skimage import transform
from skimage import io
import os

parser = argparse.ArgumentParser()
parser.add_argument('--attn_map', type=str)
parser.add_argument('--category', type=str)
parser.add_argument('--thresh', type=float, default=0.5)
parser.add_argument('--out_fol', type=str)
args = parser.parse_args()


if(args.attn_map is None):
    print("Provide --attn_map")
    exit()
if(args.out_fol is None):
   print("Provide --out_fol")
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

maps, names, labels = [], [], []

with open(args.attn_map, "rb") as fp:      
    while(True):
        try:
            gen_maps = pickle.load(fp)
            batch_names = pickle.load(fp)    
            preds = pickle.load(fp)        
            for i in range(gen_maps.shape[0]):
                maps.append(gen_maps[i, :])
                names.append(batch_names[i])
                labels.append(preds[i])
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

            # Read ground truth
            orig_images = []
            for name in names:
                with Image.open(name.decode('UTF-8')) as im:
                    orig_images.append(np.asarray(im))

            try:
                os.makedirs(args.out_fol)
            except:
                pass

            tot_c = 0
            correct_c = 0
            tot_inter = 0
            tot_union = 0
            for i in range(num_imgs):
                if((i+1) % 10 == 0):
                    print(f"Image {i+1} has label {labels[i]}")
                prediction, gt_orig = maps[i], ground_imgs[i]
                prediction -= np.amin(prediction)
                prediction = prediction / np.amax(prediction)
                # print(prediction)
                # if(i == 1):
                #     print(gt)
                #     print(np.sum(gt))
                #     print(np.where(gt == 255))
                #     print(gt.shape[0]*gt.shape[1])
                #     exit()
                # print(prediction.shape)
                # print(gt.shape)
                # exit()
                prediction = transform.resize(prediction, gt_orig.shape)
                # Write attn map to file
                # save_img = prediction * 255.0/np.amax(prediction)
                # io.imsave(args.out_fol+f'/pred_{i}.png', prediction)
                # print(np.amax(gt))

                # exit()
                # gt = []
                if(np.amax(gt_orig) == 255):
                    gt = (gt_orig / 255).copy()
                else:
                    gt = gt_orig.copy()
                io.imsave(args.out_fol+f'/gt_{i}.png', np.ndarray.astype(gt*255, np.uint8))
                io.imsave(args.out_fol+f'/orig_{i}.png', orig_images[i])
                try:
                    threshold = filters.threshold_otsu(prediction)
                except:
                    continue
                io.imsave(args.out_fol+f'/map_pred_{i}.png', np.ndarray.astype(prediction*255, np.uint8))
                # print("GAP EXTREMES", np.amax(prediction), np.amin(prediction))
                prediction[prediction < threshold] = 0
                prediction[prediction >= threshold] = 1
                # print("GAP EXTREMES", np.amax(prediction), np.amin(prediction))
                io.imsave(args.out_fol+f'/pred_{i}.png', np.ndarray.astype(prediction*255, np.uint8))
                # prediction = transform.resize(prediction, gt.shape)
                # prediction[prediction < threshold] = 0
                # prediction[prediction >= threshold] = 1
                # print(f"Pred sum = {np.sum(prediction)}, threshold={threshold}")
                # IOU
                gt_scale = gt
                # print("MAX",np.amax(gt))
                intersection = np.sum(prediction * gt_scale)
                union = np.sum(prediction) + np.sum(gt_scale) - intersection
                iou = intersection/union
                tot_inter += intersection
                tot_union += union
                if(iou >= 0.5):
                    correct_c += 1
                tot_c += 1
            print(f"Score = {correct_c/tot_c}")
            print(f"Total jaccard = {tot_inter/tot_union}")
            exit()
            

