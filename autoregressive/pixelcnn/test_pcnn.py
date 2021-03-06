import argparse
import os
import torch
import operator
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn import metrics

import data.dataset
import autoregressive.pixelcnn.model
import utils.process
import utils.metrics

def test(pcnn, testset, batch_size, directory):
    """
    Evaluate the given model
    Args:
        model (torch.nn.Module): Trained model
        testset (torch.utils.data.Dataset): Test set
        batch_size (int): Mini batch size
        directory (str): Directory to save results
    """

    with open(os.path.join(directory, 'output'), 'w') as outf:
        pass

    likelihood = []
    groundtruth = []
    dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    likelihood_distributions = {'normal': [], 'abnormal': []}
    items = {}

    dist = torch.nn.PairwiseDistance(p=2, eps=1e-06)
    reconstruction_distributions = {'normal': [], 'abnormal': []}
    reconstruction_error = []

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
        _, argmax = torch.max(probs, dim=1)
        probs = probs * onehot_lbl
        probs = torch.sum(probs, 1)

        #Draw probabilities images
        if i_batch == 0:
            for i in range(probs.size(0)):
                plt.clf()
                imgprobs = probs[i]
                imgprobs = imgprobs.data.cpu().numpy()
                plt.imshow(imgprobs)
                name = sample['name'][i]
                if '.png' in name:
                    name = name[:-4]
                plt.savefig(os.path.join(directory, 'plots', 'imgprobs_{}.svg'.format(name)), bbox_inches='tight')

        #Compute log likelihood
        probs = torch.log(probs)
        probs = probs.view((-1, 64 * 64))
        probs = torch.sum(probs, dim=1)
        probs = probs.data.cpu().numpy().tolist()
        likelihood += probs

        with open(os.path.join(directory, 'output'), 'a') as outf:
            for elt in range(len(probs)):
                outf.write('{}\t{}\n'.format(sample['complete_name'][elt], probs[elt]))

        #Reconstruction error
        reconstruction = utils.process.preprocess(argmax)
        tmp = utils.metrics.per_image_error(dist, reconstruction.float(), img)
        reconstruction_error += tmp.data.cpu().numpy().tolist()

    likelihood = np.array(likelihood)
    likelihood[likelihood == -np.inf] = likelihood[likelihood != -np.inf].min() #Remove -inf

    for i in range(len(likelihood)):
        items[testset[i]['complete_name']] = likelihood[i]
        if testset[i]['lbl'] == 0:
            likelihood_distributions['abnormal'].append(likelihood[i])
            reconstruction_distributions['abnormal'].append(reconstruction_error[i])
        else:
            likelihood_distributions['normal'].append(likelihood[i])
            reconstruction_distributions['normal'].append(reconstruction_error[i])

    #Sorted log likelihood
    sorted_items = sorted(items.items(), key=operator.itemgetter(1))

    #Compute AUC likelihood
    likelihood = np.array(likelihood)
    groundtruth = np.array(groundtruth)
    print(likelihood.shape, groundtruth.shape)
    fpr, tpr, thresholds = metrics.roc_curve(groundtruth, likelihood)
    auc = metrics.auc(fpr, tpr)
    print('AUC likelihood:', auc)

    #Compute AUC reconstruction
    reconstruction_error = np.array(reconstruction_error)
    fpr_r, tpr_r, thresholds_r = metrics.roc_curve(groundtruth, reconstruction_error)
    auc_r = metrics.auc(fpr_r, tpr_r)
    print('AUC reconstruction:', auc_r)

    #Get log likelihood histogram for normal and abnormal patterns
    normal_distribution = np.array(likelihood_distributions['normal'])
    abnormal_distribution = np.array(likelihood_distributions['abnormal'])
    print("Normal: mean={}, var={}, std={}".format(normal_distribution.mean(), normal_distribution.var(), normal_distribution.std()))
    print("Anomaly: mean={}, var={}, std={}".format(abnormal_distribution.mean(), abnormal_distribution.var(), abnormal_distribution.std()))
    hist_n, _ = np.histogram(normal_distribution, bins=50, range=[abnormal_distribution.min(), normal_distribution.max()])
    hist_a, _ = np.histogram(abnormal_distribution, bins=50, range=[abnormal_distribution.min(), normal_distribution.max()])
    minima = np.minimum(hist_n, hist_a)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_a))
    print('Intersection: {}'.format(intersection))

    plt.clf()
    weights = np.ones_like(normal_distribution)/(len(normal_distribution))
    plt.hist(normal_distribution, bins=100, alpha=0.5, weights=weights, label='Normal', color='blue')
    weights = np.ones_like(abnormal_distribution)/(len(normal_distribution))
    x2, bins2, p2 = plt.hist(abnormal_distribution, bins=100, alpha=0.5, weights=weights, label='Abnormal', color='red')
    for item2 in p2:
        item2.set_height(item2.get_height()/sum(x2))
    plt.xlabel('Log likelihood')
    plt.ylabel('Normalized number of images')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(directory, 'plots', 'loglikelihood_hist'), format='svg', bbox_inches='tight')

    #Plot time series log likelihood
    x = np.array([i for i in range(len(groundtruth))])
    likelihood = (likelihood - likelihood.min()) / (likelihood.max() - likelihood.min())
    plt.clf()
    plt.plot(x, groundtruth, '--', c='green', label='Groundtruth')
    plt.plot(x, likelihood, '-', c='red', label='Norm. log likelihood')
    plt.xlabel('Frames')
    plt.ylabel('Likelihood')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(directory, 'plots', 'abnormal_score_series'), format='svg', bbox_inches='tight')

    #Get reconstruction errors histogram for normal and abnormal patterns
    normal_distribution = np.array(reconstruction_distributions['normal'])
    abnormal_distribution = np.array(reconstruction_distributions['abnormal'])
    print("Normal (reconstruction): mean={}, var={}, std={}".format(normal_distribution.mean(), normal_distribution.var(), normal_distribution.std()))
    print("Anomaly (reconstruction): mean={}, var={}, std={}".format(abnormal_distribution.mean(), abnormal_distribution.var(), abnormal_distribution.std()))
    hist_n, _ = np.histogram(normal_distribution, bins=50, range=[abnormal_distribution.min(), normal_distribution.max()])
    hist_a, _ = np.histogram(abnormal_distribution, bins=50, range=[abnormal_distribution.min(), normal_distribution.max()])
    minima = np.minimum(hist_n, hist_a)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_a))
    print('Intersection (reconstruction): {}'.format(intersection))

    plt.clf()
    weights = np.ones_like(normal_distribution)/(len(normal_distribution))
    plt.hist(normal_distribution, bins=100, alpha=0.5, weights=weights, label='Normal', color='blue')
    weights = np.ones_like(abnormal_distribution)/(len(normal_distribution))
    x2, bins2, p2 = plt.hist(abnormal_distribution, bins=100, alpha=0.5, weights=weights, label='Abnormal', color='red')
    for item2 in p2:
        item2.set_height(item2.get_height()/sum(x2))
    plt.xlabel('L2 norm')
    plt.ylabel('Normalized number of images')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(directory, 'plots', 'reconstruction_hist'), format='svg', bbox_inches='tight')

    #AUC Mix
    norm_likelihood = (likelihood - likelihood.min()) / (likelihood.max() - likelihood.min())
    norm_reconstruction = (reconstruction_error - reconstruction_error.min()) / (reconstruction_error.max() - reconstruction_error.min())
    norm_reconstruction = norm_reconstruction.reshape(-1)
    print(norm_likelihood.shape, norm_reconstruction.shape)
    for beta in range(11):
        b = beta / 10
        score = ((1 - b) * norm_likelihood) + (b * norm_reconstruction)
        fpr_mix, tpr_mix, thresholds_mix = metrics.roc_curve(groundtruth, score)
        auc_mix = metrics.auc(fpr_mix, tpr_mix)
        print('AUC mix (beta={})'.format(b), auc_mix)

    #Store results into a file
    with open(os.path.join(directory, 'evaluation'), 'w') as f:
        f.write('AUC:\t{}\n'.format(auc))
        f.write('Normal:\tmean={},\tvar={},\tstd={}\n'.format(normal_distribution.mean(), normal_distribution.var(), normal_distribution.std()))
        f.write("Anomaly:\tmean={},\tvar={},\tstd={}\n".format(abnormal_distribution.mean(), abnormal_distribution.var(), abnormal_distribution.std()))
        f.write('Intersection:\t{}\t'.format(intersection))
        for s in sorted_items:
            f.write('{}\n'.format(s))

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
        pcnn = autoregressive.pixelcnn.model.PixelCNN(int(hp['f']), int(hp['n']), int(hp['d']))
    pcnn.cuda()
    print(pcnn)

    if args.model == '':
        model = os.path.join(args.directory, 'serial', 'best_model')
    else:
        model = args.model
    #Load the trained model
    pcnn.load_state_dict(torch.load(model))

    testset = data.dataset.VideoDataset(hp['testset'], hp['root_dir'], 'L', hp['image_size'])

    #Evaluate the model
    test(pcnn, testset, args.batch_size, args.directory)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #Test arguments
    parser.add_argument('-m', dest='model', type=str, default='', help='Serialized model')
    parser.add_argument('--tes', dest='testset', type=str, default='data/summaries/umn_testset', help='Path to the testset summary')
    parser.add_argument('--rd', dest='root_dir', type=str, default='/home/scom/data/umn64', help='Path to the images')
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
    parser.add_argument('--ims', dest='image_size', type=str, default='64,64,1', help='Image size')
    args = parser.parse_args()

    main(args)
