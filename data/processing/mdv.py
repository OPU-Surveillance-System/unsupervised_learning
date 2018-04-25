"""
Split and label videos from the Mini-drone video dataset (https://mmspg.epfl.ch/mini-drone)
"""

import argparse
import os
import imageio
import ntpath
import numpy as np
import xml.etree.ElementTree as et
from scipy import misc
from scipy import io
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('-d', dest='dataset', type=str, default='/home/scom/Downloads/datasets/MiniDrone', help='Path to the MDV dataset')
parser.add_argument('-t', dest='target', type=str, default='/home/scom/mdv64', help='Directory to store resulting frames')
parser.add_argument('-r', dest='resize', type=str, default='64,64', help='Specify the size at which the frames should be resized (format: heigt,width)')
parser.add_argument('-s', dest='suspicious', type=int, default=0, help='Consider suspicious scenes (loitering actions) as abnormal or not')

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

def get_actions(xml):
    root = et.parse(xml).getroot()[1][0]

    #Get number of frames
    for c in root:
        if 'file' in c.tag:
            for attr in c:
                if attr.attrib['name'] == 'NUMFRAMES':
                    nb_frame = int(attr[0].attrib['value'])

    #Get objects
    objects = [o for o in root if 'object' in o.tag]

    actions = [[] for i in range(nb_frame)]
    for o in objects:
        for features in o:
            if features.attrib['name'] == 'Action':
                for act in features:
                    framespan = act.attrib['framespan'].split(':')
                    for i in range(int(framespan[0]) - 1, int(framespan[1])):
                        actions[i].append(act.attrib['value'])

    return actions, nb_frame

abnormal_actions = ['fighting', 'picking_up', 'attacking', 'stealing', 'cycling', 'running', 'repairing', 'falling']
if args.suspicious == 1:
    abnormal_actions.append('loitering')

print('Process train set')
with open(os.path.join(args.target, 'train', 'mdv_trainset'), 'w') as fi: #Empty summary file
    pass
files = sorted([os.path.join(os.path.join(args.dataset, 'DroneProtect-training-set', f[:-4])) for f in os.listdir(os.path.join(args.dataset, 'DroneProtect-training-set')) if '.mp4' in f])
for f in tqdm(files):
    v = imageio.get_reader(f + '.mp4', 'ffmpeg')
    filename = ntpath.basename(f)[:-4]
    labels, nb_frames = get_actions(f + '.xgtf')
    with open(os.path.join(args.target, 'train', 'mdv_trainset'), 'a') as fi:
        for frame in range(nb_frames):
            try: #Sometimes a RuntimeError occurs while fetching the last frame
                if len(list(set(labels[frame]).intersection(abnormal_actions))) == 0:  #Check if abnormal actions are present in the frame
                    f_data = v.get_data(frame)
                    if resize is not None:
                        misc.imsave('{}_{}.png'.format(os.path.join(args.target, 'train', filename), frame), misc.imresize(f_data, (resize[0], resize[1])))
                    else:
                        misc.imsave('{}_{}.png'.format(os.path.join(args.target, 'train', filename), frame), f_data)
                    fi.write('{}_{}.png\t1\n'.format(os.path.join('train', filename), frame))
            except RuntimeError:
                pass

print('Process test set')
with open(os.path.join(args.target, 'train', 'mdv_testset'), 'w') as fi: #Empty summary file
    pass
files = sorted([os.path.join(os.path.join(args.dataset, 'DroneProtect-testing-set', f[:-4])) for f in os.listdir(os.path.join(args.dataset, 'DroneProtect-testing-set')) if '.mp4' in f])
for f in tqdm(files):
    v = imageio.get_reader(f + '.mp4', 'ffmpeg')
    filename = ntpath.basename(f)[:-4]
    labels, nb_frames = get_actions(f + '.xgtf')
    with open(os.path.join(args.target, 'test', 'mdv_testset'), 'a') as fi:
        for frame in range(nb_frames):
            try: #Sometimes a RuntimeError occurs while fetching the last frame
                f_data = v.get_data(frame)
                if resize is not None:
                    misc.imsave('{}_{}.png'.format(os.path.join(args.target, 'test', filename), frame), misc.imresize(f_data, (resize[0], resize[1])))
                else:
                    misc.imsave('{}_{}.png'.format(os.path.join(args.target, 'test', filename), frame), f_data)
                if len(list(set(labels[frame]).intersection(abnormal_actions))) != 0: #Check if abnormal actions are present in the frame
                    l = 0
                else:
                    l = 1
                fi.write('{}_{}.png\t{}\n'.format(os.path.join('test', filename), frame, l))
            except RuntimeError:
                pass
