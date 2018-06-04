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

def test(pcnn, testset, pixel, batch_size, directory):
    """
    Evaluate the given model
    Args:
        model (torch.nn.Module): Trained model
        testset (torch.utils.data.Dataset): Test set
        pixel (list of tuple of int): pixel coordinate
        batch_size (int): Mini batch size
        directory (str): Directory to save results
    """

    dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    hist = [[] for p in range(len(pixel))]

    #Process the testset
    for i_batch, sample in enumerate(tqdm(dataloader)):

        img = Variable(sample['img'], volatile=True).float().cuda()

        #Compute pixel probabilities
        probs = pcnn(img)[0]
        probs = torch.nn.functional.softmax(probs, dim=1)
        _, argmax = torch.max(probs, dim=1)

        print(sample['name'][0])
        for i in range(img.size(0)):
            if sample['name'][i] == 'test/16_737.png':
                distribution = probs[i].data.cpu().numpy()

    print(distribution)

        # for p in range(len(pixel)):
        #     for b in range(img.size(0)):
        #         hist[p].append(argmax.data.cpu().numpy()[b, pixel[p][0], pixel[p][1]])

    # for p in range(len(pixel)):
    #     histogram = np.array(hist[p])
    #     plt.clf()
    #     plt.hist(histogram, bins=128, alpha=0.5, color='blue')
    #     plt.xlabel('Intensities')
    #     plt.ylabel('Frequency')
    #     plt.xlim(0,255)
    #     plt.savefig(os.path.join(directory, 'activations_{}_{}.svg'.format(pixel[p][0], pixel[p][1])), bbox_inches='tight')
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

    pixels = [(int(p.split(',')[0]), int(p.split(',')[1])) for p in args.pixels.split(';')]

    if args.model == '':
        model = os.path.join(args.directory, 'serial', 'best_model')
    else:
        model = args.model
    #Load the trained model
    pcnn.load_state_dict(torch.load(model))

    testset = data.dataset.VideoDataset(hp['testset'], hp['root_dir'], 'L', hp['image_size'])

    #Evaluate the model
    test(pcnn, testset, pixels, int(hp['batch_size']), args.directory)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #Test arguments
    parser.add_argument('-m', dest='model', type=str, default='', help='Serialized model')
    parser.add_argument('--tes', dest='testset', type=str, default='data/summaries/umn_testset', help='Path to the testset summary')
    parser.add_argument('--rd', dest='root_dir', type=str, default='/home/scom/data/umn64', help='Path to the images')
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('-p', dest='pixels', type=str, default='0,0;0,63;63,0;63,63;5,30', help='Path to the images')
    parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
    parser.add_argument('--ims', dest='image_size', type=str, default='64,64,1', help='Image size')
    args = parser.parse_args()

    main(args)
