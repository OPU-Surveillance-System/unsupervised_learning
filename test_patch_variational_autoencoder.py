import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn import metrics

import dataset
import models.patch_variational_autoencoder
import utils.metrics
import utils.plot
import utils.process

def sample_z(mu, sigma):
    # Using reparameterization trick to sample from a gaussian
    eps = Variable(torch.randn(mu.size(0), mu.size(1))).float().cuda()
    z = mu + torch.exp(sigma / 2) * eps

    return z

def test(model, testset, batch_size, directory):
    """
    Evaluate the given model
    Args:
        model (tuple of torch.nn.Module): Trained encoder and decoder
        testset (torch.utils.data.Dataset): Test set
        batch_size (int): Mini batch size
        directory (str): Directory to save results
    """

    encoder, decoder = model
    errors = {'normal':[], 'abnormal':[]}
    answer = []
    groundtruth = []
    dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    dist = torch.nn.PairwiseDistance(p=2, eps=1e-06)

    #Process the testset
    for i_batch, sample in enumerate(tqdm(dataloader)):
        inputs = Variable(sample['img'].float().cuda())
        mu, sigma = encoder(inputs)
        z = sample_z(mu, sigma)
        logits = decoder(z).view(-1, 3, args.ips, args.ips)
        logits = torch.nn.sigmoid(logits)
        e = utils.metrics.per_image_error(dist, logits.contiguous(), inputs.contiguous())
        e = e.cpu().data.numpy().tolist()
        answer += e
        groundtruth += sample['lbl'].cpu().numpy().tolist()
        for i in range(len(sample['lbl'])):
            if sample['lbl'][i] == 0:
                errors['normal'].append(e[i])
            else:
                errors['abnormal'].append(e[i])

    #Get histograms of reconstruction error for normal and abnormal patterns
    normal_distribution = np.array(errors['normal'])
    abnormal_distribution = np.array(errors['abnormal'])
    print("Normal: mean={}, var={}, std={}".format(normal_distribution.mean(), normal_distribution.var(), normal_distribution.std()))
    print("Anomaly: mean={}, var={}, std={}".format(abnormal_distribution.mean(), abnormal_distribution.var(), abnormal_distribution.std()))
    hist_n, _ = np.histogram(normal_distribution, bins=50, range=[normal_distribution.min(), abnormal_distribution.max()])
    hist_a, _ = np.histogram(abnormal_distribution, bins=50, range=[normal_distribution.min(), abnormal_distribution.max()])
    minima = np.minimum(hist_n, hist_a)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_a))
    utils.plot.plot_reconstruction_hist(normal_distribution, abnormal_distribution, os.path.join(directory, 'plots', 'reconstruction_hist.svg'))
    print('Intersection: {}'.format(intersection))

    #Compute AUC
    fpr, tpr, thresholds = metrics.roc_curve(groundtruth, answer)
    auc = metrics.auc(fpr, tpr)
    utils.plot.plot_auc(fpr, tpr, auc, os.path.join(directory, 'plots', 'auc.svg'))
    print('AUC: {}'.format(auc))

    with open(os.path.join(directory, 'results'), 'w') as f:
        f.write('Normal: mean={}, var={}, std={}\n'.format(normal_distribution.mean(), normal_distribution.var(), normal_distribution.std()))
        f.write('Abnormal: mean={}, var={}, std={}\n'.format(abnormal_distribution.mean(), abnormal_distribution.var(), abnormal_distribution.std()))
        f.write('Intersection: {}\n'.format(intersection))
        f.write('AUC: {}\n'.format(auc))

    return 0

def main(args):
    """
    Evaluates a serialized model
    """

    #Create directories
    if not os.path.exists(os.path.join(args.directory, 'plots')):
        os.makedirs(os.path.join(args.directory, 'plots'))

    encoder = models.patch_variational_autoencoder.Encoder(args.nb_f, args.nb_l, args.nb_b, args.latent_size, args.patch)
    decoder = models.patch_variational_autoencoder.Decoder(encoder.last_map_dim, args.nb_l, args.nb_b, args.latent_size)
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    model = (encoder, decoder)
    print(encoder)
    print(decoder)
    #Load the trained model
    if args.model == -1:
        saved_enc = 'best_encoder'
        saved_dec = 'best_decoder'
    else:
        saved_enc = 'encoder_{}'.format(args.model)
        saved_dec = 'decoder_{}'.format(args.model)
    encoder.load_state_dict(torch.load(os.path.join(args.directory, 'serial', saved_enc)))
    decoder.load_state_dict(torch.load(os.path.join(args.directory, 'serial', saved_dec)))
    model = (encoder, decoder)

    testset = dataset.VideoDataset(args.testset, args.root_dir)

    #Evaluate the model
    test(model, testset, args.batch_size, args.directory)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #Test arguments
    parser.add_argument('-m', dest='model', type=int, default=-1, help='Serialized model')
    parser.add_argument('--tes', dest='testset', type=str, default='data/umn/testset', help='Path to the testset summary')
    parser.add_argument('--rd', dest='root_dir', type=str, default='/datasets', help='Path to the images')
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
    #Model arguments
    parser.add_argument('-f', dest='nb_f', type=int, default=16, help='Number of filters in the first downsampling block')
    parser.add_argument('-l', dest='nb_l', type=int, default=1, help='Number of convolutinal layers per block')
    parser.add_argument('-b', dest='nb_b', type=int, default=2, help='Number of upsampling blocks')
    parser.add_argument('-z', dest='latent_size', type=int, default=512, help='Size of latent codes')
    parser.add_argument('-i', dest='ips', type=int, default=256, help='Image height (assume width = height)')
    parser.add_argument('-p', dest='patch', type=int, default=32, help='Image patch size')
    args = parser.parse_args()

    main(args)
