import argparse
import os
import torch

import data.dataset
import autoregressive.pixelcnn.model
import autoregressive.pixelcnn.train_pcnn as train
import autoregressive.pixelcnn.test_pcnn as test

parser = argparse.ArgumentParser(description='')
#Training arguments
parser.add_argument('--trs', dest='trainset', type=str, default='data/summaries/umn_normal_trainset', help='Path to the trainset summary')
parser.add_argument('--val', dest='valset', type=str, default='data/summaries/umn_testset', help='Path to the testset summary')
parser.add_argument('--tes', dest='testset', type=str, default='data/summaries/umn_testset', help='Path to the testset summary')
parser.add_argument('--rd', dest='root_dir', type=str, default='home/scom/data/umn64', help='Path to the images')
parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--ep', dest='epoch', type=int, default=100, help='Number of training epochs')
parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
parser.add_argument('--ims', dest='image_size', type=str, default='64,64,1', help='Image size')
parser.add_argument('-v', dest='validation_size', type=float, default=0.3, help='Ratio of testset\'s elements used for validation')
parser.add_argument('-p', dest='patience', type=int, default=100, help='Early stopping max patience')
#Model arguments
parser.add_argument('-f', dest='f', type=int, default=128, help='Number of hidden features')
parser.add_argument('-d', dest='d', type=int, default=32, help='Number of top layer features')
parser.add_argument('-n', dest='n', type=int, default=15, help='Number of residual blocks')
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
if not os.path.exists(os.path.join(args.directory, 'generation')):
    os.makedirs(os.path.join(args.directory, 'generation'))
if not os.path.exists(os.path.join(args.directory, 'logs')):
    os.makedirs(os.path.join(args.directory, 'logs'))
if not os.path.exists(os.path.join(args.directory, 'plots')):
    os.makedirs(os.path.join(args.directory, 'plots'))

#Write arguments in a file
d = vars(args)
with open(os.path.join(args.directory, 'hyper-parameters'), 'w') as f:
    for k in d.keys():
        f.write('{}:{}\n'.format(k, d[k]))

#Variables
pcnn = autoregressive.pixelcnn.model.PixelCNN(args.f, args.n, args.d)
pcnn = pcnn.cuda()
print(pcnn)
optimizer = torch.optim.Adam(pcnn.parameters(), args.learning_rate)
ims = [int(s) for s in args.image_size.split(',')]

trainset = data.dataset.VideoDataset(args.trainset, args.root_dir, 'L', args.image_size)
valset = data.dataset.VideoDataset(args.valset, args.root_dir, 'L', args.image_size, val=args.validation_size)
testset = data.dataset.VideoDataset(args.testset, args.root_dir, 'L', args.image_size)
datasets = [trainset, valset]

#Train the model and save it
print('Start training.')
best_model = train.train(pcnn, optimizer, datasets, args.epoch, args.batch_size, args.patience, ims, args.directory)
print('Training complete.')

#Evaluate the model
print('Start evaluation.')
test.test(best_model, testset, args.batch_size, args.directory)
print('Evaluation complete.')
