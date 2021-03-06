import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn import metrics

import autoregressive.pixelcnn.mnist.pixelcnn_mnist
import utils.plot
import data.dataset

def test(pcnn, testset, batch_size, directory):
    """
    Evaluate the given model
    Args:
        model (torch.nn.Module): Trained model
        testset (torch.utils.data.Dataset): Test set
        batch_size (int): Mini batch size
        directory (str): Directory to save results
    """
    torch.set_printoptions(threshold=5000)
    likelihood = []
    groundtruth = []
    dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)
    dist = torch.nn.PairwiseDistance(p=2, eps=1e-06)
    likelihood_distributions = {'mnist': [], 'alphabet': []}

    #Process the testset
    for i_batch, sample in enumerate(tqdm(dataloader)):
        # if i_batch > 1:
        #     break
        img = Variable(sample[0], volatile=True).cuda()
        lbl = Variable(img.data[:, 0] * 255, volatile=True).long().cuda()
        lbl = torch.unsqueeze(lbl, 1)
        groundtruth += [1 for g in range(img.size(0))]
        onehot_lbl = torch.FloatTensor(img.size(0), 256, 28, 28).zero_().cuda()
        onehot_lbl = Variable(onehot_lbl.scatter_(1, lbl.data, 1))

        probs = pcnn(img)[0]
        probs = torch.nn.functional.softmax(probs, dim=1)
        probs = probs * onehot_lbl
        probs = torch.sum(probs, 1)
        if i_batch == 0:
            for i in range(probs.size(0)):
                plt.clf()
                imgprobs = probs[i]
                imgprobs = imgprobs.data.cpu().numpy()
                plt.imshow(imgprobs)
                plt.savefig(os.path.join(directory, 'plots', 'normal_imgprobs_{}.svg'.format(i)), format='svg', bbox_inches='tight')
        probs = torch.log(probs) #* -1
        probs = probs.view((-1, 28 * 28))
        probs = torch.sum(probs, dim=1)
        probs = probs.data.cpu().numpy().tolist()
        likelihood += probs
        likelihood_distributions['mnist'] += probs

    alphabet_dir = '/home/scom/data/alphabet_mnist'
    alphabetset = data.dataset.VideoDataset('data/summaries/alphabet_mnist', alphabet_dir, 'L', '28,28,1')
    dataloader = DataLoader(alphabetset, batch_size=batch_size, shuffle=True, num_workers=4)
    items = {}
    #Process the testset
    for i_batch, sample in enumerate(tqdm(dataloader)):
        img = Variable(sample['img'], volatile=True).float().cuda()
        lbl = Variable(img.data[:, 0] * 255, volatile=True).long().cuda()
        lbl = torch.unsqueeze(lbl, 1)
        groundtruth += sample['lbl'].numpy().tolist()
        onehot_lbl = torch.FloatTensor(img.size(0), 256, 28, 28).zero_().cuda()
        onehot_lbl = Variable(onehot_lbl.scatter_(1, lbl.data, 1))

        probs = pcnn(img)[0]
        probs = torch.nn.functional.softmax(probs, dim=1)
        probs = probs * onehot_lbl
        probs = torch.sum(probs, 1)
        for i in range(probs.size(0)):
            plt.clf()
            imgprobs = probs[i]
            imgprobs = imgprobs.data.cpu().numpy()
            plt.imshow(imgprobs)
            plt.savefig(os.path.join(directory, 'plots', 'imgprobs_{}.svg'.format(i)), format='svg', bbox_inches='tight')
        probs = torch.log(probs) #* -1
        probs = probs.view((-1, 28 * 28))
        probs = torch.sum(probs, dim=1)
        probs = probs.data.cpu().numpy().tolist()
        likelihood += probs
        likelihood_distributions['alphabet'] += probs

        for n in range(img.size(0)):
            items[sample['name'][n]] = probs[n]

    print(items)

    fpr, tpr, thresholds = metrics.roc_curve(groundtruth, likelihood)
    auc = metrics.auc(fpr, tpr)
    print('AUC:', auc)

    mnist_distribution = np.array(likelihood_distributions['mnist'])
    alphabet_distribution = np.array(likelihood_distributions['alphabet'])
    print("Normal: mean={}, var={}, std={}".format(mnist_distribution.mean(), mnist_distribution.var(), mnist_distribution.std()))
    print("Anomaly: mean={}, var={}, std={}".format(alphabet_distribution.mean(), alphabet_distribution.var(), alphabet_distribution.std()))
    hist_m, _ = np.histogram(mnist_distribution, bins=50, range=[mnist_distribution.min(), alphabet_distribution.max()])
    hist_a, _ = np.histogram(alphabet_distribution, bins=50, range=[mnist_distribution.min(), alphabet_distribution.max()])
    minima = np.minimum(hist_m, hist_a)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_a))
    utils.plot.plot_mnist_likelihood_hist(mnist_distribution, alphabet_distribution, os.path.join(directory, 'plots', 'loglikelihood_hist.svg'))
    print('Intersection: {}'.format(intersection))

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
        pcnn = autoregressive.pixelcnn.mnist.pixelcnn_mnist.PixelCNN(int(hp['f']), int(hp['n']), int(hp['d']))
    pcnn.cuda()
    print(pcnn)
    #Load the trained model
    pcnn.load_state_dict(torch.load(args.model))

    testset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())

    #Evaluate the model
    test(pcnn, testset, args.batch_size, args.directory)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #Test arguments
    parser.add_argument('-m', dest='model', type=str, default='', help='Serialized model')
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
    args = parser.parse_args()

    main(args)
