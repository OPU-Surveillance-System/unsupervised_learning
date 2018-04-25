"""
Split and label videos from the CWFID dataset (https://github.com/cwfid/dataset)
"""

import argparse
import os
import yaml
import numpy as np
from scipy import misc
from scipy import io
from tqdm import tqdm
from sklearn.feature_extraction import image

parser = argparse.ArgumentParser()

parser.add_argument('-d', dest='dataset', type=str, default='/home/scom/Downloads/dataset-1.0', help='Path to the CWFID dataset')
parser.add_argument('-t', dest='target', type=str, default='/home/scom/cwfid64', help='Directory to store resulting frames')
parser.add_argument('-r', dest='resize', default='1024,1024', help='Specify the size at which the frames should be resized (format: heigt,width)')
parser.add_argument('-p', dest='patch', default='64,64', help='Specify the patch size (format: heigt,width)')

args = parser.parse_args()

if not os.path.exists(args.target):
    os.makedirs(args.target)
if not os.path.exists(os.path.join(args.target, 'train')):
    os.makedirs(os.path.join(args.target, 'train'))
if not os.path.exists(os.path.join(args.target, 'test')):
    os.makedirs(os.path.join(args.target, 'test'))

if args.resize is not '':
    resize = [int(dim) for dim in args.resize.split(',')]
else:
    resize = None

if args.patch is not '':
    patch = [int(dim) for dim in args.patch.split(',')]
else:
    patch = None

print('Get train and test sets elements')
with open('/home/scom/Downloads/dataset-1.0/train_test_split.yaml', 'r') as f:
    content = yaml.load(f)
    train_elements = content['train']
    test_elements = content['test']

print('Process train set')
with open(os.path.join(args.target, 'cwfid_trainset'), 'w') as fi: #Empty summary file
    pass
for t in train_elements:
    index = str(t)
    index = '0' * (3 - len(index)) + index #Image name includes one or more "0" before the image's index
    inimg = misc.imread(os.path.join(args.dataset, 'images', '{}_image.png'.format(index)), 'L')
    gtimg = misc.imread(os.path.join(args.dataset, 'annotations', '{}_annotation.png'.format(index)))
    if resize:
        inimg = misc.imresize(inimg, (resize[0], resize[1]))
        gtimg = misc.imresize(gtimg, (resize[0], resize[1]))
    gtimg = gtimg[:, :, 1]
    with open(os.path.join(args.target, 'cwfid_trainset'), 'a') as fi:
        count = 0
        for y in range(inimg.shape[0] // patch[0]):
            for x in range(inimg.shape[1] // patch[1]):
                l = not np.any(gtimg[y * patch[0]:(y * patch[0]) + patch[0], x * patch[1]:(x * patch[1]) + patch[1]])
                if l:
                    misc.imsave('{}_{}.png'.format(os.path.join(args.target, 'train', str(t)), count), inimg[y * patch[0]:(y * patch[0]) + patch[0], x * patch[1]:(x * patch[1]) + patch[1]])
                    fi.write('{}_{}.png\t1\n'.format(os.path.join('train', str(t)), count))
                count += 1

print('Process test set')
with open(os.path.join(args.target, 'cwfid_testset'), 'w') as fi: #Empty summary file
    pass
for t in test_elements:
    index = str(t)
    index = '0' * (3 - len(index)) + index #Image name includes one or more "0" before the image's index
    inimg = misc.imread(os.path.join(args.dataset, 'images', '{}_image.png'.format(index)), 'L')
    gtimg = misc.imread(os.path.join(args.dataset, 'annotations', '{}_annotation.png'.format(index)))
    if resize:
        inimg = misc.imresize(inimg, (resize[0], resize[1]))
        gtimg = misc.imresize(gtimg, (resize[0], resize[1]))
    gtimg = gtimg[:, :, 1]
    with open(os.path.join(args.target, 'cwfid_testset'), 'a') as fi:
        count = 0
        for y in range(inimg.shape[0] // patch[0]):
            for x in range(inimg.shape[1] // patch[1]):
                l = int(not np.any(gtimg[y * patch[0]:(y * patch[0]) + patch[0], x * patch[1]:(x * patch[1]) + patch[1]]))
                misc.imsave('{}_{}.png'.format(os.path.join(args.target, 'test', str(t)), count), inimg[y * patch[0]:(y * patch[0]) + patch[0], x * patch[1]:(x * patch[1]) + patch[1]])
                fi.write('{}_{}.png\t{}\n'.format(os.path.join('test', str(t)), count, l))
                count += 1
