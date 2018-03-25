import argparse
import os
import torch
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

    errors = {'normal':[], 'abnormal':[]}
    answer = []
    groundtruth = []
    dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    dist = torch.nn.PairwiseDistance(p=2, eps=1e-06)

    #Process the testset
    for i_batch, sample in enumerate(tqdm(dataloader)):
        img = Variable(sample['img'], volatile=True).float().cuda()
        lbl = Variable(img.data[:, 0] * 255, volatile=True).long().cuda()
        lbl = torch.unsqueeze(lbl, 1)
        print('lbl', lbl.shape)
        output = pcnn(img)[1]
        print('output', output.shape)
        onehot_lbl = torch.FloatTensor(batch_size, 256, 64, 64).zero_().cuda()
        onehot_lbl = Variable(onehot_lbl.scatter_(1, lbl.data, 1))
        print('onehot', onehot_lbl.shape)
        merge = output * onehot_lbl
        print('merge', merge.shape)
        merge = torch.sum(merge, 1)
        merge = torch.unsqueeze(merge, 1)
        print(merge[0][0])
        print('sum', merge.shape)
        prob = torch.prod(merge, 1)
        print('prod', prob.shape, prob)
    #     e = utils.metrics.per_image_error(dist, logits.contiguous(), inputs.contiguous())
    #     e = e.cpu().data.numpy().tolist()
    #     answer += e
    #     groundtruth += sample['lbl'].cpu().numpy().tolist()
    #     for i in range(len(sample['lbl'])):
    #         if sample['lbl'][i] == 0:
    #             errors['normal'].append(e[i])
    #         else:
    #             errors['abnormal'].append(e[i])
    #
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
    #
    # #Compute AUC
    # fpr, tpr, thresholds = metrics.roc_curve(groundtruth, answer)
    # auc = metrics.auc(fpr, tpr)
    # utils.plot.plot_auc(fpr, tpr, auc, os.path.join(directory, 'plots', 'auc.svg'))
    # print('AUC: {}'.format(auc))
    #
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
