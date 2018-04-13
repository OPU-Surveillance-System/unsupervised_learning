import argparse
import os
import torch
import operator
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

    likelihood = []
    groundtruth = []
    dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    likelihood_distributions = {'normal': [], 'abnormal': []}
    items = {}

    #Process the testset
    for i_batch, sample in enumerate(tqdm(dataloader)):
        groundtruth += sample['lbl'].numpy().tolist()

        img = Variable(sample['img'], volatile=True).float().cuda()
        lbl = Variable(img.data[:, 0] * 255, volatile=True).long().cuda()
        lbl = torch.unsqueeze(lbl, 1)
        onehot_lbl = torch.FloatTensor(img.size(0), 256, 64, 64).zero_().cuda()
        onehot_lbl = Variable(onehot_lbl.scatter_(1, lbl.data, 1))

        #Compute pixel probabilities
        probs = pcnn(img)[0]
        probs = torch.nn.functional.softmax(probs, dim=1)
        probs = probs * onehot_lbl
        probs = torch.sum(probs, 1)

        #Draw probabilities images
        for i in range(probs.size(0)):
            plt.clf()
            imgprobs = probs[i]
            imgprobs = imgprobs.data.cpu().numpy()
            plt.imshow(imgprobs)
            plt.savefig(os.path.join(directory, 'plots', 'imgprobs_{}.svg'.format(sample['name'][i]), bbox_inches='tight'))

        #Compute log likelihood
        probs = torch.log(probs)
        probs = probs.view((-1, 64, 64))
        probs = torch.sum(probs, dim=1)
        probs = probs.data.cpu().numpy().tolist()
        likelihood += probs

        for i in range(img.size(0)):
            items[sample['name'][i]] = probs[i]

        for i in range(len(sample['lbl'])):
            if sample['lbl'][i] == 0:
                likelihood_distributions['abnormal'].append(probs[i])
            else:
                likelihood_distributions['normal'].append(probs[i])

    #Print sorted log likelihood
    sorted_items = sorted(items.items(), key=operator.itemgetter(1))
    print(sorted_items)

    #Fix infinite log likelihood
    likelihood = np.array(likelihood)
    likelihood[likelihood == -np.inf] = likelihood[likelihood != -np.inf].min()

    #Compute AUC
    fpr, tpr, thresholds = metrics.roc_curve(groundtruth, likelihood)
    auc = metrics.auc(fpr, tpr)
    print('AUC:', auc)

    #Get log likelihood histogram for normal and abnormal patterns
    normal_distribution = np.array(likelihood_distributions['normal'])
    abnormal_distribution = np.array(likelihood_distributions['abnormal'])
    print("Normal: mean={}, var={}, std={}".format(normal_distribution.mean(), normal_distribution.var(), normal_distribution.std()))
    print("Anomaly: mean={}, var={}, std={}".format(abnormal_distribution.mean(), abnormal_distribution.var(), abnormal_distribution.std()))
    hist_n, _ = np.histogram(normal_distribution, bins=50, range=[abnormal_distribution.min(), normal_distribution.max()])
    hist_a, _ = np.histogram(abnormal_distribution, bins=50, range=[abnormal_distribution.min(), normal_distribution.max()])
    minima = np.minimum(hist_n, hist_a)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_a))
    utils.plot.plot_likelihood_hist(normal_distribution, abnormal_distribution, os.path.join(directory, 'plots', 'loglikelihood_hist.svg'))
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
        if hp['bn'] == '0':
            bn = False
        else:
            bn = True
        pcnn = autoregressive.pixelcnn.PixelCNN(int(hp['f']), int(hp['n']), int(hp['d']), bn)
    pcnn.cuda()
    print(pcnn)

    if args.model == '':
        model = os.path.join(args.directory, 'serial', 'best_model')
    else:
        model = args.model
    #Load the trained model
    pcnn.load_state_dict(torch.load(model))

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
