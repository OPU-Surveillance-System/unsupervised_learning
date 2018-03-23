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
import autoregressive.pixelcnn
import utils.metrics
import utils.plot
import utils.process

def train(pcnn, optimizer, trainset, testset, epoch, batch_size, directory):
    """
    """

    phase = ('train', 'test')
    sets = {'train':trainset, 'test':testset}

    writer = SummaryWriter(os.path.join(directory, 'logs'))

    for e in range(epoch):
        running_loss = 0

        for p in phase:
            pcnn.train(p == 'train')

            dataloader = DataLoader(sets[p], batch_size=batch_size, shuffle=False, num_workers=4)

            for i_batch, sample in enumerate(tqdm(dataloader)):
                if i_batch > 0:
                    break
                optimizer.zero_grad()
                img = Variable(sample[0], volatile=(p == 'test')).cuda()
                lbl = Variable(sample[0] * 255, volatile=(p == 'test')).long().cuda()

                logits = pcnn(img)[0]

                loss = torch.nn.functional.cross_entropy(logits.view((-1, 256)), lbl.view((-1)))
                running_loss += loss.data[0]
                if p == 'train':
                    loss.backward()
                    optimizer.step()

            epoch_loss = running_loss / (i_batch + 1)
            writer.add_scalar('learning_curve/{}'.format(p), epoch_loss, e)
            print('Epoch {} ({}): loss = {}'.format(e, p, epoch_loss))

            if p == 'test':
                synthetic = torch.zeros(16, 1, 28, 28).cuda()
                for i in tqdm(range(28)):
                    for j in range(28):
                        probs = pcnn(Variable(synthetic, volatile=True))[0]
                        probs = torch.nn.functional.softmax(probs[:, :, i, j]).data
                        synthetic[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.

                synthetic = synthetic.cpu().numpy()
                synthetic = np.moveaxis(synthetic, 1, -1)
                synthetic = np.reshape(synthetic, (4 * 28, 4 * 28))
                plt.clf()
                plt.imshow(synthetic)
                plt.savefig(os.path.join(directory, 'generation', '{}.svg'.format(e)), format='svg', bbox_inches='tight')

            if p == 'train':
                logits = logits.permute(0, 2, 3, 1)
                probs = torch.nn.functional.softmax(logits, dim=3)
                argmax = torch.max(probs, 3)[1]
                argmax = argmax.data.cpu().numpy()
                argmax = np.reshape(argmax, (batch_size, 28, 28))[0:4]
                argmax = np.reshape(argmax, (2 * 28, 2 * 28))
                plt.clf()
                plt.imshow(argmax)
                plt.savefig(os.path.join(directory, 'reconstruction_train', '{}.svg'.format(e)), format='svg', bbox_inches='tight')

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
    pcnn = autoregressive.pixelcnn.PixelCNN(args.f, args.n, args.d)
    pcnn = pcnn.cuda()
    print(pcnn)
    optimizer = torch.optim.Adam(pcnn.parameters(), args.learning_rate)

    trainset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    #Train the model and save it
    best_model = train(pcnn, optimizer, trainset, testset, args.epoch, args.batch_size, args.directory)
    # torch.save(best_model.state_dict(), os.path.join(args.directory, 'serial', 'best_model'))

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #Training arguments
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--ep', dest='epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
    #Model arguments
    parser.add_argument('-f', dest='f', type=int, default=128, help='Number of hidden features')
    parser.add_argument('-d', dest='d', type=int, default=32, help='Number of top layer features')
    parser.add_argument('-n', dest='n', type=int, default=15, help='Number of residual blocks')
    args = parser.parse_args()

    main(args)