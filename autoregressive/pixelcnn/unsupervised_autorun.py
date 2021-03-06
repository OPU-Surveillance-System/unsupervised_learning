import argparse
import os
import torch

import data.dataset
import data.unsupervised_dataset
import autoregressive.pixelcnn.model
import autoregressive.pixelcnn.train_pcnn as train
import autoregressive.pixelcnn.test_pcnn as test

parser = argparse.ArgumentParser(description='')
#Training arguments
parser.add_argument('--ntrs', dest='n_trainset', type=str, default='data/summaries/umn_normal_trainset', help='Path to the normal trainset summary')
parser.add_argument('--atrs', dest='a_trainset', type=str, default='data/summaries/umn_abnormal_trainset', help='Path to the abnormal trainset summary')
parser.add_argument('--tes', dest='testset', type=str, default='data/summaries/umn_testset', help='Path to the testset summary')
parser.add_argument('--rd', dest='root_dir', type=str, default='home/scom/data/umn64', help='Path to the images')
parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--ep', dest='epoch', type=int, default=200, help='Number of training epochs')
parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
parser.add_argument('--ims', dest='image_size', type=str, default='64,64,1', help='Image size')
parser.add_argument('-p', dest='patience', type=int, default=100, help='Early stopping max patience')

#Model arguments
parser.add_argument('-f', dest='f', type=int, default=128, help='Number of hidden features')
parser.add_argument('-d', dest='d', type=int, default=32, help='Number of top layer features')
parser.add_argument('-n', dest='n', type=int, default=15, help='Number of residual blocks')
args = parser.parse_args()

# ratios = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 40, 50]
ratios = [30, 40, 50, 60]
for r in ratios:
    for t in range(10):
        directory = os.path.join(args.directory, 'r_{}_{}'.format(r, t))
        #Create directories if it don't exists
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(os.path.join(directory, 'serial')):
            os.makedirs(os.path.join(directory, 'serial'))
        if not os.path.exists(os.path.join(directory, 'reconstruction_train')):
            os.makedirs(os.path.join(directory, 'reconstruction_train'))
        if not os.path.exists(os.path.join(directory, 'reconstruction_test')):
            os.makedirs(os.path.join(directory, 'reconstruction_test'))
        if not os.path.exists(os.path.join(directory, 'generation')):
            os.makedirs(os.path.join(directory, 'generation'))
        if not os.path.exists(os.path.join(directory, 'logs')):
            os.makedirs(os.path.join(directory, 'logs'))
        if not os.path.exists(os.path.join(directory, 'plots')):
            os.makedirs(os.path.join(directory, 'plots'))

        #Write arguments in a file
        d = vars(args)
        with open(os.path.join(directory, 'hyper-parameters'), 'w') as f:
            for k in d.keys():
                f.write('{}:{}\n'.format(k, d[k]))

        #Variables
        pcnn = autoregressive.pixelcnn.model.PixelCNN(args.f, args.n, args.d)
        pcnn = pcnn.cuda()
        print(pcnn)
        optimizer = torch.optim.Adam(pcnn.parameters(), args.learning_rate)
        ims = [int(s) for s in args.image_size.split(',')]

        trainset = data.unsupervised_dataset.UnsupervisedDataset(args.n_trainset, args.a_trainset, args.root_dir, r)
        testset = data.dataset.VideoDataset(args.testset, args.root_dir, 'L', args.image_size)
        datasets = [trainset, testset]

        #Train the model and save it
        print('Start training.')
        best_model = train.train(pcnn, optimizer, datasets, args.epoch, args.batch_size, args.patience, 0.0, ims, directory, generation=False)
        print('Training complete.')

        #Evaluate the model
        print('Start evaluation.')
        test.test(best_model, testset, args.batch_size, directory)
        print('Evaluation complete.')
