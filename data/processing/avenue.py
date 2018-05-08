"""
Split and label videos from the Avenue dataset (http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
"""

import random
import argparse
import os
import imageio
import ntpath
import numpy as np
from scipy import misc
from scipy import io
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('-d', dest='dataset', type=str, default='/home/scom/Downloads/Avenue Dataset', help='Path to the avenue dataset')
parser.add_argument('-t', dest='target', type=str, default='/home/scom/avenue64', help='Directory to store resulting frames')
parser.add_argument('-r', dest='resize', type=str, default='64,64', help='Specify the size at which the frames should be resized (format: heigt,width)')
parser.add_argument('-s', dest='avenue17', type=int, default=0, help='If 1 the testset produced is Avenue17 as proposed in https://arxiv.org/abs/1709.09121')
parser.add_argument('-e', dest='avenueext', type=int, default=0, help='If 1 the testset include and label (as abnormal) the frame removed in Avenue17')

args = parser.parse_args()

if args.avenue17 == 1 and args.avenueext == 1:
    raise Exception('Avenue17 and Avenueext cannot be activated together')

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

if args.avenue17 == 1:
    addon = '17'
elif args.avenueext == 1:
    addon = 'ext'
else:
    addon = ''

avenue17 = ['01', '02', '08', '09', '10']

print('Process train set')
with open(os.path.join(args.target, 'avenue_trainset'), 'w') as fi: #Empty summary file
    pass
files = sorted([os.path.join(os.path.join(args.dataset, 'training_videos', f)) for f in os.listdir(os.path.join(args.dataset, 'training_videos'))])
for f in tqdm(files):
    v = imageio.get_reader(f, 'ffmpeg')
    nb_frames = v.get_meta_data()['nframes']
    filename = ntpath.basename(f)[:-4]
    with open(os.path.join(args.target, 'avenue_trainset'), 'a') as fi:
        for frame in range(nb_frames):
            try: #Sometimes a RuntimeError occurs while fetching the last frame
                f_data = v.get_data(frame)
                if resize is not None:
                    misc.imsave('{}_{}.png'.format(os.path.join(args.target, 'train', filename), frame), misc.imresize(f_data, (resize[0], resize[1])))
                else:
                    misc.imsave('{}_{}.png'.format(os.path.join(args.target, 'train', filename), frame), f_data)
            except RuntimeError:
                pass
            fi.write('{}_{}.png\t1\n'.format(os.path.join('train', filename), frame))

print('Process test set')
valset = []
with open(os.path.join(args.target, 'avenue_testset' + addon), 'w') as fi:
    pass
files = sorted([os.path.join(os.path.join(args.dataset, 'testing_videos', f)) for f in os.listdir(os.path.join(args.dataset, 'testing_videos'))])
for f in tqdm(files):
    if args.avenue17 == 1 and any(clip in f for clip in avenue17):
        continue
    v = imageio.get_reader(f, 'ffmpeg')
    nb_frames = v.get_meta_data()['nframes']
    filename = ntpath.basename(f)[:-4]
    labels = io.loadmat(os.path.join(args.dataset, 'ground_truth_demo', 'testing_label_mask', '{}_label.mat'.format(int(filename))))['volLabel'][0]
    with open(os.path.join(args.target, 'avenue_testset' + addon), 'a') as fi:
        for frame in range(nb_frames):
            try: #Sometimes a RuntimeError occurs while fetching the last frame
                f_data = v.get_data(frame)
                if resize is not None:
                    misc.imsave('{}_{}.png'.format(os.path.join(args.target, 'test', filename), frame), misc.imresize(f_data, (resize[0], resize[1])))
                else:
                    misc.imsave('{}_{}.png'.format(os.path.join(args.target, 'test', filename), frame), f_data)
            except RuntimeError:
                pass
            if args.avenueext == 1 and any(clip in f for clip in avenue17):
                l = 0
            else:
                l = int(not np.any(labels[frame]))
            fi.write('{}_{}.png\t{}\n'.format(os.path.join('test', filename), frame, l))
            valset.append('{}_{}.png\t{}'.format(os.path.join('test', filename), frame, l))
random.seed(a=1204)
random.shuffle(valset)
with open(os.path.join(args.target, 'avenue_valset' + addon), 'w') as fi:
    for v in valset:
        fi.write('{}\n'.format(v))
