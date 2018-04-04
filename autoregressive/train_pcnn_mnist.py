import argparse
import os
import torch
import copy
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from sklearn import metrics
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import dataset
import autoregressive.pixelcnn_mnist
import utils.metrics
import utils.plot
import utils.process

def train(pcnn, optimizer, trainset, testset, epoch, batch_size, directory, translate=False):
    """
    """

    phase = ('train', 'test')
    sets = {'train':trainset, 'test':testset}

    writer = SummaryWriter(os.path.join(directory, 'logs'))

    best_auc = 0.0
    best_model = copy.deepcopy(pcnn)

    #translation = transforms.RandomAffine(translate=(0.25, 0.25))

    for e in range(epoch):

        likelihood = []
        groundtruth = []

        for p in phase:
            running_loss = 0
            pcnn.train(p == 'train')

            dataloader = DataLoader(sets[p], batch_size=batch_size, shuffle=True, num_workers=4)

            for i_batch, sample in enumerate(tqdm(dataloader)):
                optimizer.zero_grad()
                img = Variable(sample[0], volatile=(p == 'test')).cuda()
                # if p == 'train' and translate:
                #     img = translation(img)
                lbl = Variable(img.data[:, 0] * 255, volatile=(p == 'test')).long().cuda()

                logits = pcnn(img)[0]

                loss = torch.nn.functional.cross_entropy(logits, lbl)
                running_loss += loss.data[0]
                if p == 'train':
                    loss.backward()
                    optimizer.step()
                if p == 'test':
                    lbl = torch.unsqueeze(lbl, 1)
                    groundtruth += [0 for g in range(img.size(0))]
                    onehot_lbl = torch.FloatTensor(img.size(0), 256, 28, 28).zero_().cuda()
                    onehot_lbl = Variable(onehot_lbl.scatter_(1, lbl.data, 1))

                    probs = torch.nn.functional.softmax(logits, dim=1)
                    probs = probs * onehot_lbl
                    probs = torch.sum(probs, 1)
                    probs = torch.log(probs) * -1
                    probs = probs.view((-1, 28 * 28))
                    probs = torch.sum(probs, dim=1)
                    probs = probs.data.cpu().numpy().tolist()
                    likelihood += probs

            epoch_loss = loss.data[0] / (i_batch + 1)

            if p == 'test':
                alphabet_dir = '/home/scom/data/alphabet_mnist'
                alphabetset = dataset.VideoDataset('data/alphabet_mnist', alphabet_dir, 'L', '28,28,1')
                dataloader = DataLoader(alphabetset, batch_size=batch_size, shuffle=True, num_workers=4)
                items = {}
                #Process the testset
                for i_batch, sample in enumerate(tqdm(dataloader)):
                    a_img = Variable(sample['img'], volatile=True).float().cuda()
                    a_lbl = Variable(a_img.data[:, 0] * 255, volatile=True).long().cuda()
                    a_lbl = torch.unsqueeze(a_lbl, 1)
                    groundtruth += sample['lbl'].numpy().tolist()
                    a_onehot_lbl = torch.FloatTensor(a_img.size(0), 256, 28, 28).zero_().cuda()
                    a_onehot_lbl = Variable(a_onehot_lbl.scatter_(1, a_lbl.data, 1))

                    a_probs = pcnn(img)[0]
                    a_probs = torch.nn.functional.softmax(a_probs, dim=1)
                    a_probs = a_probs * a_onehot_lbl
                    a_probs = torch.sum(a_probs, 1)
                    a_probs = torch.log(a_probs) * -1
                    a_probs = a_probs.view((-1, 28 * 28))
                    a_probs = torch.sum(a_probs, dim=1)
                    a_probs = a_probs.data.cpu().numpy().tolist()
                    likelihood += a_probs

                fpr, tpr, thresholds = metrics.roc_curve(groundtruth, likelihood)
                auc = metrics.auc(fpr, tpr)
            else:
                auc = 0.0

            writer.add_scalar('learning_curve/{}'.format(p), epoch_loss, e)
            writer.add_scalar('auc/{}'.format(p), auc, e)
            print('Epoch {} ({}): loss = {}, AUC = {}'.format(e, p, epoch_loss, auc))

            if p == 'test' and e % 10 == 0:
                synthetic = torch.zeros(16, 1, 28, 28).cuda()
                for i in tqdm(range(28)):
                    for j in range(28):
                        probs = pcnn(Variable(synthetic, volatile=True))[0]
                        probs = torch.nn.functional.softmax(probs[:, :, i, j]).data
                        synthetic[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.

                synthetic = synthetic.cpu().numpy()
                synthetic = np.reshape(synthetic, (4, 4, 28, 28))
                synthetic = np.swapaxes(synthetic, 1, 2)
                synthetic = np.reshape(synthetic, (28 * 4, 28 * 4))
                plt.clf()
                plt.imshow(synthetic)
                plt.savefig(os.path.join(directory, 'generation', '{}.svg'.format(e)), format='svg', bbox_inches='tight')

                #torch.save(pcnn.state_dict(), os.path.join(directory, 'serial', 'model_{}'.format(e)))

                if auc > best_auc:
                    best_model = copy.deepcopy(pcnn)
                    torch.save(pcnn.state_dict(), os.path.join(directory, 'serial', 'best_model'.format(e)))
                    best_auc = auc

            #Plot reconstructions
            logits = logits.permute(0, 2, 3, 1)
            probs = torch.nn.functional.softmax(logits, dim=3)
            argmax = torch.max(probs, 3)[1]
            argmax = argmax.data.cpu().numpy()
            lbl = lbl.data.cpu().numpy()
            nb_img = min(argmax.shape[0], 4)
            lbl = np.reshape(lbl, (-1, 28, 28))[0:nb_img]
            argmax = np.reshape(argmax, (-1, 28, 28))[0:nb_img]

            plt.clf()
            lbl = np.reshape(lbl, (1, nb_img, 28, 28))
            lbl = np.swapaxes(lbl, 1, 2)
            lbl = np.reshape(lbl, (28, nb_img * 28))
            ax = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
            ax.imshow(lbl)
            argmax = np.reshape(argmax, (1, nb_img, 28, 28))
            argmax = np.swapaxes(argmax, 1, 2)
            argmax = np.reshape(argmax, (28, nb_img * 28))
            ax = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)
            ax.imshow(argmax)
            plt.savefig(os.path.join(directory, 'reconstruction_{}'.format(p), '{}.svg'.format(e)), format='svg', bbox_inches='tight')

    return best_model

def main(args):
    """
    Train an autoencoder and save it
    """

    #Create directories if it don't exists
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    if not os.path.exists(os.path.join(args.directory, 'serial')):
        os.makedirs(os.path.join(args.directory, 'serial'))
    if not os.path.exists(os.path.join(args.directory, 'reconstruction_train')):
        os.makedirs(os.path.join(args.directory, 'reconstruction_train'))
    if not os.path.exists(os.path.join(args.directory, 'reconstruction_test')):
        os.makedirs(os.path.join(args.directory, 'reconstruction_test'))
    if not os.path.exists(os.path.join(args.directory, 'generation')):
        os.makedirs(os.path.join(args.directory, 'generation'))
    if not os.path.exists(os.path.join(args.directory, 'logs')):
        os.makedirs(os.path.join(args.directory, 'logs'))

    #Write arguments in a file
    d = vars(args)
    with open(os.path.join(args.directory, 'hyper-parameters'), 'w') as f:
        for k in d.keys():
            f.write('{}:{}\n'.format(k, d[k]))

    #Variables
    pcnn = autoregressive.pixelcnn_mnist.PixelCNN(args.f, args.n, args.d)
    pcnn = pcnn.cuda()
    print(pcnn)
    optimizer = torch.optim.Adam(pcnn.parameters(), args.learning_rate)

    trainset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    #Train the model and save it
    if args.translate == 0:
        translate = False
    else:
        translate = True
    best_model = train(pcnn, optimizer, trainset, testset, args.epoch, args.batch_size, args.directory, translate)
    #torch.save(best_model.state_dict(), os.path.join(args.directory, 'serial', 'best_model'))

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #Training arguments
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--ep', dest='epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
    parser.add_argument('-t', dest='translate', type=int, default=0, help='Translate training images')
    #Model arguments
    parser.add_argument('-f', dest='f', type=int, default=128, help='Number of hidden features')
    parser.add_argument('-d', dest='d', type=int, default=32, help='Number of top layer features')
    parser.add_argument('-n', dest='n', type=int, default=15, help='Number of residual blocks')
    args = parser.parse_args()

    main(args)
