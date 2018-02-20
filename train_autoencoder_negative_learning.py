import argparse
import os
import torch
import copy
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn import metrics
from tensorboardX import SummaryWriter

import dataset
import models.autoencoder
import utils.metrics
import utils.plot
import utils.process

def train(model, loss_function, optimizer, n_trainset, a_trainset, testset, negative_weight, epoch, batch_size, directory):
    """
    Train a model and log the process
    Args:
        model (torch.nn.Module): Model to train (autoencoder)
        loss_function (torch.optim.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        n_trainset (torch.utils.data.Dataset): Training set of normal patterns
        a_trainset (torch.utils.data.Dataset): Training set of abnormal patterns
        testset (torch.utils.data.Dataset): Test set
        negative_weight (float): Weight for the negative loss
        epoch (int): Number of training epochs
        batch_size (int): Mini batch size
        directory (str): Directory to store the logs
    """

    phase = ('n_train', 'a_train', 'test')
    datasets = {'n_train': n_trainset, 'a_train': a_trainset, 'test': testset}
    dist = torch.nn.PairwiseDistance(p=2, eps=1e-06)
    best_auc = 0
    best_model = copy.deepcopy(model)
    writer = SummaryWriter(os.path.join(directory, 'logs'))

    for e in range(epoch):
        print('Epoch {}'.format(e))
        for p in phase:
            if p == 'n_train' or p == 'a_train':
                model.train()
            else:
                labels = []
                errors = []
                model.eval()
            running_loss = 0
            dataloader = DataLoader(datasets[p], batch_size=batch_size, shuffle=True, num_workers=4)
            for i_batch, sample in enumerate(tqdm(dataloader)):
                model.zero_grad()
                inputs = Variable(sample['img'].float().cuda())
                logits, pred = model(inputs)
                loss = loss_function(logits, inputs)
                if p == 'n_train' or p == 'a_train':
                    if p == 'a_train':
                        loss = - negative_weight * loss
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data[0]
                if p == 'test':
                    tmp = utils.metrics.per_image_error(dist, logits, inputs)
                    errors += tmp.data.cpu().numpy().tolist()
                    labels += sample['lbl'].numpy().tolist()

            #Compute AUC
            if p == 'test':
                fpr, tpr, thresholds = metrics.roc_curve(labels, errors)
                auc = metrics.auc(fpr, tpr)
            else:
                auc = 0
            epoch_loss = running_loss / len(datasets[p])
            writer.add_scalar('learning_curve/{}'.format(p), epoch_loss, e)
            print('{} -- Loss: {} AUC: {}'.format(p, epoch_loss, auc))
            #Memorize model with the best AUC, save model every 10 epochs and save some examples of reconstructed images
            if p == 'test':
                writer.add_scalar('auc', auc, e)
                if auc > best_auc:
                    best_auc = auc
                    best_model = copy.deepcopy(model)
                if e % 10 == 0:
                    #Save model
                    torch.save(model.state_dict(), os.path.join(directory, 'serial', 'model_{}'.format(e)))

                    #Plot example of reconstructed images
                    pred = utils.process.deprocess(pred)
                    pred = pred.data.cpu().numpy()
                    pred = np.rollaxis(pred, 1, 4)
                    inputs = utils.process.deprocess(inputs)
                    inputs = inputs.data.cpu().numpy()
                    inputs = np.rollaxis(inputs, 1, 4)
                    utils.plot.plot_reconstruction_images(inputs, pred, os.path.join(directory, 'example_reconstruction', 'epoch_{}.svg'.format(e)))
    writer.close()

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
    if not os.path.exists(os.path.join(args.directory, 'example_reconstruction')):
        os.makedirs(os.path.join(args.directory, 'example_reconstruction'))
    if not os.path.exists(os.path.join(args.directory, 'logs')):
        os.makedirs(os.path.join(args.directory, 'logs'))

    #Write arguments in a file
    d = vars(args)
    with open(os.path.join(args.directory, 'hyper-parameters'), 'w') as f:
        for k in d.keys():
            f.write('{}:{}\n'.format(k, d[k]))

    #Variables
    ae = models.autoencoder.Autoencoder(args.nb_f, args.nb_l, args.nb_b, args.dense, args.ips, args.act)
    ae = ae.cuda()
    print(ae)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), args.learning_rate)

    n_trainset = dataset.VideoDataset(args.normal_trainset, args.root_dir)
    a_trainset = dataset.VideoDataset(args.abnormal_trainset, args.root_dir)
    testset = dataset.VideoDataset(args.testset, args.root_dir)

    #Train the model and save it
    best_model = train(ae, loss_function, optimizer, n_trainset, a_trainset, testset, args.negative_weight, args.epoch, args.batch_size, args.directory)
    torch.save(best_model.state_dict(), os.path.join(args.directory, 'serial', 'best_model'))

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #Training arguments
    parser.add_argument('--trsn', dest='normal_trainset', type=str, default='data/umn_normal_trainset', help='Path to the trainset summary of normal patterns')
    parser.add_argument('--trsa', dest='abnormal_trainset', type=str, default='data/umn_abnormal_trainset', help='Path to the trainset summary of abnormal patterns')
    parser.add_argument('--tes', dest='testset', type=str, default='data/umn_testset', help='Path to the testset summary')
    parser.add_argument('--rd', dest='root_dir', type=str, default='/datasets', help='Path to the images')
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--nw', dest='negative_weight', type=float, default=0.1, help='Negative loss weight')
    parser.add_argument('--ep', dest='epoch', type=int, default=100, help='Number of training epochs')
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
