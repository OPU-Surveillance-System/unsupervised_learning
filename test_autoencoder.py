import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn import metrics

import dataset
import models.autoencoder
import utils.metrics
import utils.plot
import utils.process

def test(model, testset, batch_size, directory):
    """
    Evaluate the given model
    Args:
        model (torch.nn.Module): Trained model
        testset (torch.utils.data.Dataset): Test set
        batch_size (int): Mini batch size
        directory (str): Directory to save results
    """

    errors = {'normal':[], 'abnormal':[]}
    answer = []
    groundtruth = []
    dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    dist = torch.nn.PairwiseDistance(p=2, eps=1e-06)

    #Process the testset
    for i_batch, sample in enumerate(tqdm(dataloader)):
        inputs = Variable(utils.process.preprocess(sample['img'].float().cuda()))
        pred = model(inputs)[1]
        e = utils.metrics.per_image_error(dist, pred.contiguous(), inputs.contiguous())
        e = e.cpu().data.numpy().tolist()
        answer += e
        print(sample['lbl'])
        groundtruth += sample['lbl'].cpu().data.numpy().tolist()
        for i in range(len(sample['lbl'])):
            if sample['lbl'][i] == 0:
                errors['normal'].append(e[i])
            else:
                sample['abnormal'].append(e[i])

    #Get histograms of reconstruction error for normal and abnormal patterns
    normal_distribution = np.array(errors['normal'])
    abnormal_distribution = np.array(errors['abnormal'])
    print("Normal: mean={}, var={}, std={}".format(normal_distribution.mean(), normal_distribution.var(), normal_distribution.std()))
    print("Anomaly: mean={}, var={}, std={}".format(abnormal_distribution.mean(), abnormal_distribution.var(), abnormal_distribution.std()))
    hist_n, _ = np.histogram(normal_distribution, bins=100, range=[normal_distribution.min(), abnormal_distribution.max()])
    hist_a, _ = np.histogram(abnormal_distribution, bins=100, range=[normal_distribution.min(), abnormal_distribution.max()])
    minima = np.minimum(hist_n, hist_a)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_a))
    utils.plot.plot_reconstruction_hist(normal_distribution, abnormal_distribution, os.path.join(directory, 'plots', 'reconstruction_hist.svg'))
    print('Intersection: {}'.format(intersection))

    #Compute AUC
    fpr, tpr, thresholds = metrics.roc_curve(groundtruth, answer)
    auc = metrics.auc(fpr, tpr)
    utils.plot.plot_auc(fpr, tpr, auc, os.path.join(directory, 'plots', 'auc.svg'))


    return 0

def main(args):
    """
    Evaluates a serialized model
    """

    #Create a model with the hyper-parameters used during training
    if os.path.exists(os.path.join(args.directory, 'hyper-parameters')):
        with open(os.path.join(args.directory, 'hyper-parameters'), 'r') as f:
            hp = f.read().split('\n')[:-1]
        hp = {e.split(':')[0]:e.split(':')[1] for e in hp}
        if hp['dense'] == 'None':
            hp['dense'] = None
        else:
            hp['dense'] = int(hp['dense'])
        ae = models.autoencoder.Autoencoder(int(hp['nb_f']), int(hp['nb_l']), int(hp['nb_b']), hp['dense'], int(hp['ips']), hp['act'])
    #Use hyper-parameters passed as arguments
    else:
        ae = models.autoencoder.Autoencoder(args.nb_f, args.nb_l, args.nb_b, args.dense, args.ips, args.act)
    ae.cuda()
    print(ae)
    #Load the trained model
    ae.load_state_dict(torch.load(args.model))

    testset = dataset.VideoDataset(args.testset, args.root_dir)

    #Evaluate the model
    test(ae, testset, args.batch_size, args.directory)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #Test arguments
    parser.add_argument('-m', dest='model', type=str, default='', help='Serialized model')
    parser.add_argument('--tes', dest='testset', type=str, default='data/umn/testset', help='Path to the testset summary')
    parser.add_argument('--rd', dest='root_dir', type=str, default='/datasets', help='Path to the images')
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
    #Model arguments
    parser.add_argument('-f', dest='nb_f', type=int, default=16, help='Number of filters in the first downsampling block')
    parser.add_argument('-l', dest='nb_l', type=int, default=1, help='Number of convolutinal layers per block')
    parser.add_argument('-b', dest='nb_b', type=int, default=2, help='Number of upsampling blocks')
    parser.add_argument('-d', dest='dense', type=int, default=None, help='Number of neurons in the middle denser layer (if None: no dense layer)')
    parser.add_argument('-i', dest='ips', type=int, default=256, help='Image height (assume width = height)')
    parser.add_argument('-a', dest='act', type=str, default='selu', help='Non linear activation (selu or relu)')
    args = parser.parse_args()

    main(args)
