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
import overfit.autoencoder
import utils.metrics
import utils.plot
import utils.process

def train(model, optimizer, trainset, testset, epoch, batch_size, noise_ratio, directory):
    """
    Train a model and log the process
    Args:
        model (torch.nn.Module): Model to train (autoencoder)
        loss_function (torch.optim.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        trainset (torch.utils.data.Dataset): Training set
        testset (torch.utils.data.Dataset): Test set
        epoch (int): Number of training epochs
        batch_size (int): Mini batch size
        directory (str): Directory to store the logs
    """

    phase = ('train', 'test')
    datasets = {'train': trainset, 'test': testset}
    dist = torch.nn.PairwiseDistance(p=2, eps=1e-06)
    best_auc = 0
    best_model = copy.deepcopy(model)
    writer = SummaryWriter(os.path.join(directory, 'logs'))

    for e in range(epoch):
        print('Epoch {}'.format(e))
        for p in phase:
            if p == 'train':
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
                groundtruth = Variable(sample['img'].float().cuda())
                if p == 'train':
                    noise =  torch.randn(inputs.size()) * noise_ratio
                    inputs = inputs + noise
                logits = model(inputs)
                loss = torch.nn.functional.mse_loss(logits, inputs)
                if p == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data[0]
                if p == 'test':
                    tmp = utils.metrics.per_image_error(dist, logits, inputs)
                    errors += tmp.data.cpu().numpy().tolist()
                    labels += sample['lbl'].numpy().tolist()

            if p == 'test':
                fpr, tpr, thresholds = metrics.roc_curve(labels, errors)
                auc = metrics.auc(fpr, tpr)
            else:
                auc = 0
            epoch_loss = running_loss / len(datasets[p])
            writer.add_scalar('learning_curve/{}'.format(p), epoch_loss, e)
            print('{} -- Loss: {} AUC: {}'.format(p, epoch_loss, auc))
            if p == 'test':
                writer.add_scalar('auc', auc, e)
                if auc > best_auc:
                    best_auc = auc
                    best_model = copy.deepcopy(model)
                # if e % 10 == 0:
                #     #Save model
                #     torch.save(model.state_dict(), os.path.join(directory, 'serial', 'model_{}'.format(e)))

            if e % 10 == 0:
                #Plot example of reconstructed images
                pred = utils.process.deprocess(logits)
                pred = pred.data.cpu().numpy()
                pred = np.rollaxis(pred, 1, 4)
                inputs = utils.process.deprocess(inputs)
                inputs = inputs.data.cpu().numpy()
                inputs = np.rollaxis(inputs, 1, 4)
                groundtruth = utils.process.deprocess(groundtruth)
                groundtruth = groundtruth.data.cpu().numpy()
                groundtruth = np.rollaxis(groundtruth, 1, 4)
                utils.plot.plot_reconstruction_noiy_images(groundtruth, pred, inputs, os.path.join(directory, 'example_reconstruction_{}'.format(p), 'epoch_{}.svg'.format(e)))
    writer.export_scalars_to_json(os.path.join(directory, 'logs', 'scalars.json'))
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
    if not os.path.exists(os.path.join(args.directory, 'example_reconstruction_train')):
        os.makedirs(os.path.join(args.directory, 'example_reconstruction_train'))
    if not os.path.exists(os.path.join(args.directory, 'example_reconstruction_test')):
        os.makedirs(os.path.join(args.directory, 'example_reconstruction_test'))
    if not os.path.exists(os.path.join(args.directory, 'logs')):
        os.makedirs(os.path.join(args.directory, 'logs'))

    #Write arguments in a file
    d = vars(args)
    with open(os.path.join(args.directory, 'hyper-parameters'), 'w') as f:
        for k in d.keys():
            f.write('{}:{}\n'.format(k, d[k]))

    #Variables
    ae = overfit.autoencoder.Autoencoder(3, args.nb_f, args.nb_l, args.nb_b, args.dense, args.rate)
    ae = ae.cuda()
    print(ae)
    optimizer = torch.optim.Adam(ae.parameters(), args.learning_rate)

    trainset = dataset.VideoDataset(args.trainset, args.root_dir)
    testset = dataset.VideoDataset(args.testset, args.root_dir)

    #Train the model and save it
    best_model = train(ae, optimizer, trainset, testset, args.epoch, args.batch_size, args.noise, args.directory)
    # torch.save(best_model.state_dict(), os.path.join(args.directory, 'serial', 'best_model'))

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #Training arguments
    parser.add_argument('--trs', dest='trainset', type=str, default='data/umn_normal_trainset', help='Path to the trainset summary')
    parser.add_argument('--tes', dest='testset', type=str, default='data/umn_testset', help='Path to the testset summary')
    parser.add_argument('--rd', dest='root_dir', type=str, default='/datasets', help='Path to the images')
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('-n', dest='noise', type=float, default=0.02, help='Noise ratio')
    parser.add_argument('--ep', dest='epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
    #Model arguments
    parser.add_argument('-f', dest='nb_f', type=int, default=16, help='Number of filters in the first downsampling block')
    parser.add_argument('-l', dest='nb_l', type=int, default=1, help='Number of convolutinal layers per block')
    parser.add_argument('-b', dest='nb_b', type=int, default=2, help='Number of upsampling blocks')
    parser.add_argument('-d', dest='dense', type=int, default=None, help='Number of neurons in the middle denser layer (if None: no dense layer)')
    parser.add_argument('-i', dest='ips', type=int, default=256, help='Image height (assume width = height)')
    parser.add_argument('-r', dest='rate', type=float, default=0.5, help='Dropout rate')
    args = parser.parse_args()

    main(args)
