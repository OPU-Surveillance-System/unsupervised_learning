import argparse
import os
import torch
import copy
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from sklearn import metrics
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import dataset
import utils.metrics
import utils.plot
import utils.process
import autoencoder.autoencoder_mnist

parser = argparse.ArgumentParser(description='')
#Training arguments
parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--ep', dest='epoch', type=int, default=100, help='Number of training epochs')
parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
#Model arguments
parser.add_argument('-f', dest='f', type=int, default=8, help='Number of hidden features')
parser.add_argument('-b', dest='b', type=int, default=2, help='Number of blocks')
parser.add_argument('-l', dest='l', type=int, default=2, help='Number of layers per block')
parser.add_argument('-z', dest='z', type=int, default=2, help='Latent size')

args = parser.parse_args()

#Create directories if it don't exists
if not os.path.exists(args.directory):
    os.makedirs(args.directory)
if not os.path.exists(os.path.join(args.directory, 'serial')):
    os.makedirs(os.path.join(args.directory, 'serial'))
if not os.path.exists(os.path.join(args.directory, 'reconstruction_train')):
    os.makedirs(os.path.join(args.directory, 'reconstruction_train'))
if not os.path.exists(os.path.join(args.directory, 'reconstruction_test')):
    os.makedirs(os.path.join(args.directory, 'reconstruction_test'))
if not os.path.exists(os.path.join(args.directory, 'reconstruction_alphabet')):
    os.makedirs(os.path.join(args.directory, 'reconstruction_alphabet'))
if not os.path.exists(os.path.join(args.directory, 'logs')):
    os.makedirs(os.path.join(args.directory, 'logs'))

#Write arguments in a file
d = vars(args)
with open(os.path.join(args.directory, 'hyper-parameters'), 'w') as f:
    for k in d.keys():
        f.write('{}:{}\n'.format(k, d[k]))

#Variables
ae = autoencoder.autoencoder_mnist.Autoencoder(args.f, args.l, args.b, args.z)
ae = ae.cuda()
print(ae)
optimizer = torch.optim.Adam(ae.parameters(), args.learning_rate)

dist = torch.nn.PairwiseDistance(p=2, eps=1e-06)

trainset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
testset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

phase = ('train', 'test')
sets = {'train':trainset, 'test':testset}

writer = SummaryWriter(os.path.join(args.directory, 'logs'))

best_auc = 0.0
best_model = copy.deepcopy(ae)

for e in range(args.epoch):

    errors = []
    groundtruth = []

    for p in phase:
        running_loss = 0
        ae.train(p == 'train')

        dataloader = DataLoader(sets[p], batch_size=args.batch_size, shuffle=True, num_workers=4)

        for i_batch, sample in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            inputs = Variable(sample[0].float().cuda())
            logits = ae(inputs)
            loss = torch.nn.functional.mse_loss(logits, inputs)
            running_loss += loss.data[0]

            if p == 'train':
                loss.backward()
                optimizer.step()
            if p == 'test':
                tmp = utils.metrics.per_image_error(dist, logits, inputs)
                errors += tmp.data.cpu().numpy().tolist()
                groundtruth += [0 for g in range(inputs.size(0))]

        epoch_loss = running_loss / len(sets[p])

        #Plot example of reconstructed images
        pred = utils.process.deprocess(logits)
        pred = pred.data.cpu().numpy()
        pred = np.rollaxis(pred, 1, 4)
        inputs = utils.process.deprocess(inputs)
        inputs = inputs.data.cpu().numpy()
        inputs = np.rollaxis(inputs, 1, 4)
        utils.plot.plot_reconstruction_images(inputs, pred, os.path.join(args.directory, 'reconstruction_{}'.format(p), 'epoch_{}.svg'.format(e)))

        if p == 'test':
            alphabet_dir = '/home/scom/data/alphabet_mnist'
            alphabetset = dataset.VideoDataset('data/alphabet_mnist', alphabet_dir, 'L', '28,28,1')
            dataloader = DataLoader(alphabetset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            items = {}
            #Process the testset
            for i_batch, sample in enumerate(tqdm(dataloader)):
                inputs = Variable(sample['img'].float().cuda())
                logits = ae(inputs)

                tmp = utils.metrics.per_image_error(dist, logits, inputs)
                errors += tmp.data.cpu().numpy().tolist()
                groundtruth += [1 for g in range(inputs.size(0))]

            #Plot example of reconstructed images
            pred = utils.process.deprocess(logits)
            pred = pred.data.cpu().numpy()
            pred = np.rollaxis(pred, 1, 4)
            inputs = utils.process.deprocess(inputs)
            inputs = inputs.data.cpu().numpy()
            inputs = np.rollaxis(inputs, 1, 4)
            utils.plot.plot_reconstruction_images(inputs, pred, os.path.join(args.directory, 'reconstruction_alphabet', 'epoch_{}.svg'.format(e)))

            fpr, tpr, thresholds = metrics.roc_curve(groundtruth, errors)
            auc = metrics.auc(fpr, tpr)
        else:
            auc = 0.0

        writer.add_scalar('learning_curve/{}'.format(p), epoch_loss, e)
        writer.add_scalar('auc/{}'.format(p), auc, e)
        print('Epoch {} ({}): loss = {}, AUC = {}'.format(e, p, epoch_loss, auc))

        if auc > best_auc:
            best_model = copy.deepcopy(ae)
            torch.save(ae.state_dict(), os.path.join(args.directory, 'serial', 'best_model'.format(e)))
            print('Best model saved.')
            best_auc = auc

writer.export_scalars_to_json(os.path.join(args.directory, 'logs', 'scalars.json'))
writer.close()
