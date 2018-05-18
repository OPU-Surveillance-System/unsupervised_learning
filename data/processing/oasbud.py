from scipy import io
from scipy import signal
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def map255(x):
    y = -1 * x
    y = 1 - ((y - y.min()) / (y.max() - y.min()))
    y *= 255

    return y

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return img[ymin:ymax+1, xmin:xmax+1]

target = '/home/scom/oasbud'
if not os.path.exists(target):
    os.makedirs(target)

data = io.loadmat('/home/scom/Downloads/OASBUD.mat')['data'][0]
b = 0
m = 0
thresholds = (-40, -50, -60)
train = [i[0][0] for i in data if i[6][0] == 0]
random.shuffle(train)
train = train[0:30]

with open(os.path.join(target, 'oasbud_trainset'), 'w') as f:
    pass
with open(os.path.join(target, 'oasbud_testset'), 'w') as f:
    pass

for i in data:
    pid = i[0]
    rf1 = i[1]
    rf2 = i[2]
    lab = i[6]
    if pid[0] in train:
        s = 'train'
    else:
        s = 'test'

    for t in thresholds:
        a = abs(signal.hilbert(rf1)) #Get amplitude
        alog = 20 * np.log10(a/a.max()) #Log compression
        alog[alog < t] = t #Threshold dB
        image = map255(alog) #Map to [0, 255]
        image = image * i[3] #Mask lesion
        image = bbox2(image) #Crop
        image = misc.imresize(image, (64, 64))
        misc.imsave(os.path.join(target, '{}_1_{}.png'.format(pid[0], -t)), image)
        with open(os.path.join(target, 'oasbud_{}set'.format(s)), 'a') as f:
            f.write('{}_1_{}.png\t{}\n'.format(pid[0], -t, 1-lab[0][0]))

    for t in thresholds:
        a = abs(signal.hilbert(rf2)) #Get amplitude
        alog = 20 * np.log10(a/a.max()) #Log compression
        alog[alog < t] = t #Threshold dB
        image = map255(alog) #Map to [0, 255]
        image = image * i[4] #Mask lesion
        image = bbox2(image) #Crop
        image = misc.imresize(image, (64, 64))
        misc.imsave(os.path.join(target, '{}_2_{}.png'.format(pid[0], -t)), image)
        with open(os.path.join(target, 'oasbud_{}set'.format(s)), 'a') as f:
            f.write('{}_2_{}.png\t{}\n'.format(pid[0], -t, 1-lab[0][0]))
