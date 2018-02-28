import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn import metrics

import dataset
import models.patch_adversarial_autoencoder
import utils.metrics
import utils.plot
import utils.process

def test(networks, testset, batch_size, patch_size, directory):
    """
    Evaluate the given model
    Args:
        networks (list of torch.nn.Module): Trained models (encoder, decoder, discriminator)
        testset (torch.utils.data.Dataset): Test set
        batch_size (int): Mini batch size
        directory (str): Directory to save results
    """

    errors = {'normal':[], 'abnormal':[]}
    answer = []
    groundtruth = []
    reconstruction_errors = []
    discriminator_outputs = []
    dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    dist = torch.nn.PairwiseDistance(p=2, eps=1e-06)

    encoder.eval()
    decoder.eval()
    discriminator.eval()

    #Process the testset
    for i_batch, sample in enumerate(tqdm(dataloader)):
        inputs = Variable(sample['img'].float().cuda())
        latent = encoder(inputs)
        r = decoder(latent)
        r_ = utils.metrics.per_image_error(dist, reconstruction, inputs.view(-1, 3, patch_size, patch_size))
        d = torch.nn.functional.tanh(discriminator(latent))
        reconstruction_errors += r_.cpu().numpy().tolist()
        discriminator_outputs += d.cpu().numpy().tolist()
        groundtruth += sample['lbl'].cpu().numpy().tolist()

    # #Get histograms of reconstruction error for normal and abnormal patterns
    # normal_distribution = np.array(errors['normal'])
    # abnormal_distribution = np.array(errors['abnormal'])
    # print("Normal: mean={}, var={}, std={}".format(normal_distribution.mean(), normal_distribution.var(), normal_distribution.std()))
    # print("Anomaly: mean={}, var={}, std={}".format(abnormal_distribution.mean(), abnormal_distribution.var(), abnormal_distribution.std()))
    # hist_n, _ = np.histogram(normal_distribution, bins=50, range=[normal_distribution.min(), abnormal_distribution.max()])
    # hist_a, _ = np.histogram(abnormal_distribution, bins=50, range=[normal_distribution.min(), abnormal_distribution.max()])
    # minima = np.minimum(hist_n, hist_a)
    # intersection = np.true_divide(np.sum(minima), np.sum(hist_a))
    # utils.plot.plot_reconstruction_hist(normal_distribution, abnormal_distribution, os.path.join(directory, 'plots', 'reconstruction_hist.svg'))
    # print('Intersection: {}'.format(intersection))

    #Compute AUC
    alphas = np.arange(0.0, 1.5, 0.05)
    for a in range(len(alphas)):
        image_abnormal_score = utils.metrics.mean_image_abnormal_score(reconstruction_errors, discriminator_ouput, alphas[a], patch_size)
        image_abnormal_score = image_abnormal_score.cpu().numpy().tolist()
        fpr, tpr, thresholds = metrics.roc_curve(label, image_abnormal_score)
        auc = metrics.auc(fpr, tpr)
        fpr, tpr, thresholds = metrics.roc_curve(groundtruth, answer)
        auc = metrics.auc(fpr, tpr)
        utils.plot.plot_auc(fpr, tpr, auc, os.path.join(directory, 'plots', 'auc_{}.svg.'.format(alphas[a])))
        print('Alpha = {} AUC: {}'.format(alphas[a], auc))

    # with open(os.path.join(directory, 'results'), 'w') as f:
    #     f.write('Normal: mean={}, var={}, std={}\n'.format(normal_distribution.mean(), normal_distribution.var(), normal_distribution.std()))
    #     f.write('Abnormal: mean={}, var={}, std={}\n'.format(abnormal_distribution.mean(), abnormal_distribution.var(), abnormal_distribution.std()))
    #     f.write('Intersection: {}\n'.format(intersection))
    #     f.write('AUC: {}\n'.format(auc))

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
        encoder = models.patch_autoencoder.Encoder(int(hp['nb_f']), int(hp['nb_l']), int(hp['nb_b']), int(hp['latent_size']), int(hp['patch']))
        decoder = models.patch_autoencoder.Decoder(encoder.last_map_dim, int(hp['nb_l']), int(hp['nb_b']), int(hp['latent_size']))
        discriminator = models.patch_autoencoder.Discriminator(int(hp['latent_size']), hp['denses'])
    # #Use hyper-parameters passed as arguments
    # else:
    #     ae = models.patch_autoencoder.Autoencoder(args.nb_f, args.nb_l, args.nb_b, args.dense, args.ips, args.patch)
    # ae.cuda()
    print(encoder)
    print(decoder)
    print(discriminator)
    #Load the trained model
    if args.model == -1:
        se = 'best_encoder'
        sd = 'best_decoder'
        sdi = 'best_discriminator'
    else:
        se = 'encoder_'.format(args.model)
        sd = 'decoder_'.format(args.model)
        sdi = 'discriminator_'.format(args.model)
    encoder.load_state_dict(torch.load(os.path.join(args.directory, 'serial', se)))
    decoder.load_state_dict(torch.load(os.path.join(args.directory, 'serial', sd)))
    discriminator.load_state_dict(torch.load(os.path.join(args.directory, 'serial', sdi)))
    networks = [encoder, decoder, discriminator]

    testset = dataset.VideoDataset(args.testset, args.root_dir)

    #Evaluate the model
    test(networks, testset, args.batch_size, args.directory)

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
    parser.add_argument('-d', dest='dense', type=int, default=None, help='Number of neurons in the middle denser layer (if None: no dense layer)')
    parser.add_argument('-i', dest='ips', type=int, default=256, help='Image height (assume width = height)')
    parser.add_argument('-p', dest='patch', type=int, default=32, help='Image patch size')
    args = parser.parse_args()

    main(args)
