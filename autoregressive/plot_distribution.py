import argparse
import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn import metrics

import dataset
import autoregressive.pixelcnn
import utils.metrics
import utils.plot
import utils.process

def test(pcnn, testset, batch_size, directory):
    """
    Evaluate the given model
    Args:
        model (torch.nn.Module): Trained model
        testset (torch.utils.data.Dataset): Test set
        batch_size (int): Mini batch size
        directory (str): Directory to save results
    """

    answer = []
    groundtruth = []
    dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    dist = torch.nn.PairwiseDistance(p=2, eps=1e-06)

    #Process the testset
    for i_batch, sample in enumerate(tqdm(dataloader)):
        img = Variable(sample['img'], volatile=True).float().cuda()
        lbl = Variable(img.data[:, 0] * 255, volatile=True).long().cuda()
        output = pcnn(img)[1]
        output = output.permute(0, 2, 3, 1)
        output = output.data.cpu().numpy()
        img_idx = random.randint(0, img.size(0) - 1)
        row_idx = random.randint(0, img.size(2) - 1)
        col_idx = random.randint(0, img.size(3) - 1)
        utils.plot.plot_distribution(output[img_idx][row_idx][col_idx], os.path.join(directory, 'plots', '{}_{}_{}.svg'.format(sample['name'][img_idx], row_idx, col_idx)))

    return 0

def main(args):
    """
    Evaluates a serialized model
    """

    #Create directories
    if not os.path.exists(os.path.join(args.directory, 'plots')):
        os.makedirs(os.path.join(args.directory, 'plots'))

    #Create a model with the hyper-parameters used during training
    if os.path.exists(os.path.join(args.directory, 'hyper-parameters')):
        with open(os.path.join(args.directory, 'hyper-parameters'), 'r') as f:
            hp = f.read().split('\n')[:-1]
        hp = {e.split(':')[0]:e.split(':')[1] for e in hp}
        pcnn = autoregressive.pixelcnn.PixelCNN(int(hp['f']), int(hp['n']), int(hp['d']))
    pcnn.cuda()
    print(pcnn)
    #Load the trained model
    pcnn.load_state_dict(torch.load(args.model))

    testset = dataset.VideoDataset(args.testset, args.root_dir, 'L', args.image_size)

    #Evaluate the model
    test(pcnn, testset, args.batch_size, args.directory)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #Test arguments
    parser.add_argument('-m', dest='model', type=str, default='', help='Serialized model')
    parser.add_argument('--tes', dest='testset', type=str, default='data/umn/testset', help='Path to the testset summary')
    parser.add_argument('--rd', dest='root_dir', type=str, default='/datasets', help='Path to the images')
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
    parser.add_argument('--ims', dest='image_size', type=str, default='64,64,1', help='Image size')
    args = parser.parse_args()

    main(args)
