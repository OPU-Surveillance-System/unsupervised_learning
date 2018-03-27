import argparse
import os
import torch
import numpy as np
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn import metrics

import dataset
import autoregressive.pixelcnn_mnist
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
        if i_batch > 0:
            break
        img = Variable(sample[0], volatile=(p == 'test')).cuda()
        lbl = Variable(img.data[:, 0] * 255, volatile=(p == 'test')).long().cuda()

        for i in tqdm(range(28)):
            for j in range(28):
                masked = torch.zeros(img.size(0), 1, 28, 28).cuda()
                masked[:, :, 0:i, 0:j] = img[:, :, 0:i, 0:j]
                masked = masked.cpu().numpy()
                plt.clf()
                plt.imshow(masked[0].reshape((28, 28)))
                plt.savefig(os.path.join(directory, 'plots', '{}_{}.svg'.format(i, j)), format='svg', bbox_inches='tight')
        #         probs = pcnn(Variable(synthetic, volatile=True))[0]
        #         probs = torch.nn.functional.softmax(probs[:, :, i, j]).data
        #         synthetic[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.
        #
        # synthetic = synthetic.cpu().numpy()
        # synthetic = np.reshape(synthetic, (4, 4, 28, 28))
        # synthetic = np.swapaxes(synthetic, 1, 2)
        # synthetic = np.reshape(synthetic, (28 * 4, 28 * 4))
        # plt.clf()
        # plt.imshow(synthetic)
        # plt.savefig(os.path.join(directory, 'generation', '{}.svg'.format(e)), format='svg', bbox_inches='tight')
        #
        # torch.save(pcnn.state_dict(), os.path.join(directory, 'serial', 'model_{}'.format(e)))
    #     lbl = torch.unsqueeze(lbl, 1)
    #     output = pcnn(img)[1]
    #     onehot_lbl = torch.FloatTensor(img.size(0), 256, 64, 64).zero_().cuda()
    #     onehot_lbl = Variable(onehot_lbl.scatter_(1, lbl.data, 1))
    #     merge = output * onehot_lbl
    #     merge = torch.sum(merge, 1)
    #     merge = merge.view(img.size(0), -1)
    #     merge = torch.log(merge)
    #     prob = torch.sum(merge, 1)
    #     answer += prob.data.cpu().numpy().tolist()
    #     groundtruth += sample['lbl'].numpy().tolist()
    # answer = np.array(answer)
    # answer[answer == -np.inf] = -20000
    # fpr, tpr, thresholds = metrics.roc_curve(groundtruth, answer)
    # auc = metrics.auc(fpr, tpr)
    # print('AUC:', auc)

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
        pcnn = autoregressive.pixelcnn_mnist.PixelCNN(int(hp['f']), int(hp['n']), int(hp['d']))
    pcnn.cuda()
    print(pcnn)
    #Load the trained model
    pcnn.load_state_dict(torch.load(args.model))

    testset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

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
    args = parser.parse_args()

    main(args)
