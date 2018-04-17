"""
Split and label videos from the Avenue dataset (http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
"""

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
parser.add_argument('-r', dest='resize', default='64,64', help='Specify the size at which the frames should be resized (format: heigt,width)')

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

print('Process train set')
with open(os.path.join(args.target, 'train', 'avenue_trainset'), 'w') as fi: #Empty summary file
    pass
files = sorted([os.path.join(os.path.join(args.dataset, 'training_videos', f)) for f in os.listdir(os.path.join(args.dataset, 'training_videos'))])
for f in tqdm(files):
    v = imageio.get_reader(f, 'ffmpeg')
    nb_frames = v.get_meta_data()['nframes']
    filename = ntpath.basename(f)[:-4]
    with open(os.path.join(args.target, 'train', 'avenue_trainset'), 'a') as fi:
        for frame in range(nb_frames):
            try: #Sometimes a RuntimeError occurs while fetching the last frame
                f_data = v.get_data(frame)
                if resize is not None:
                    misc.imsave('{}_{}.png'.format(os.path.join(args.target, 'train', filename), frame), misc.imresize(f_data, (resize[0], resize[1])))
                else:
                    misc.imsave('{}_{}.png'.format(os.path.join(args.target, 'train', filename), frame), f_data)
            except RuntimeError:
                pass
            fi.write('{}_{}.png\t1\n'.format(os.path.join(args.target, 'train', filename), frame))

print('Process test set')
with open(os.path.join(args.target, 'test', 'avenue_testset'), 'w') as fi:
    pass
files = sorted([os.path.join(os.path.join(args.dataset, 'testing_videos', f)) for f in os.listdir(os.path.join(args.dataset, 'testing_videos'))])
for f in tqdm(files):
    v = imageio.get_reader(f, 'ffmpeg')
    nb_frames = v.get_meta_data()['nframes']
    filename = ntpath.basename(f)[:-4]
    labels = io.loadmat(os.path.join(args.dataset, 'ground_truth_demo', 'testing_label_mask', '{}_label.mat'.format(int(filename))))['volLabel'][0]
    with open(os.path.join(args.target, 'test', 'avenue_testset'), 'a') as fi:
        for frame in range(nb_frames):
            try: #Sometimes a RuntimeError occurs while fetching the last frame
                f_data = v.get_data(frame)
                if resize is not None:
                    misc.imsave('{}_{}.png'.format(os.path.join(args.target, 'test', filename), frame), misc.imresize(f_data, (resize[0], resize[1])))
                else:
                    misc.imsave('{}_{}.png'.format(os.path.join(args.target, 'test', filename), frame), f_data)
            except RuntimeError:
                pass
            l = int(not np.any(labels[frame]))
            fi.write('{}_{}.png\t{}\n'.format(os.path.join(args.target, 'test', filename), frame, l))
