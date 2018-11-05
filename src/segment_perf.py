import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pkl', type=str)
args = parser.parse_args()

if(args.pkl is None):
    print('Provide pickle')
    exit()

fp = open(args.pkl, 'rb')
m1 = pickle.load(fp)
m2 = pickle.load(fp)

im = m2[0]
print(im.shape)
print(np.sum(im[:,:]))
# print(np.sum(im[:,:,1]))
# print(np.sum(im[:,:,2]))

# print(np.all(im[:,:,0] == im[:,:,1]))
# print(np.all(im[:,:,1] == im[:,:,2]))