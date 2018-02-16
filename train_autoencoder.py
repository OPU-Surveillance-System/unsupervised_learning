import argparse
import os
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm import tqdm
from sklearn import metrics

import dataset
import models.autoencoder
import utils.metrics

def train(model, loss_function, optimizer, trainset, testset, epoch, batch_size, directory):
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
                inputs = Variable(sample['img'].float().cuda())[0]
                logits, pred = model(inputs)
                loss = loss_function(logits, inputs)
                if p == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data[0]
                if p == 'test':
                    tmp = utils.metrics.per_image_error(dist, pred, inputs)
                    errors += tmp.numpy().tolist()
                    labels += sample['lbl'].numpy().tolist()
            if p == 'test':
                fpr, tpr, thresholds = metrics.roc_curve(groundtruth, error)
                auc = metrics.auc(fpr, tpr)
            else:
                auc = 0
            epoch_loss = running_loss / len(datasets[p])
            print('{} -- Loss: {} AUC: {}'.format(p, epoch_loss, auc))

    return 0

def main(args):
    """
    Train an autoencoder and save it
    """

    #Create results directory if it doesn't exists
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
        os.makedirs(os.path.join(args.directory, 'serial'))

    #Write arguments in a file
    d = vars(args)
    with open(os.path.join(args.directory, 'hyper-parameters'), 'w') as f:
        for k in d.keys():
            f.write('{}\t:\t{}\n'.format(k, d[k]))

    #Variables
    ae = models.autoencoder.Autoencoder(args.nb_f, args.nb_l, args.nb_b, args.dense, args.ips, args.act)
    print(ae)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), args.learning_rate)

    trainset = dataset.VideoDataset(args.trainset, args.root_dir)
    testset = dataset.VideoDataset(args.testset, args.root_dir)

    #Train the model and save it
    best_model = train(ae, loss_function, optimizer, trainset, testset, args.epoch, args.batch_size, args.directory)
    #torch.save(best_model, os.path.join(args.directory, 'serial', 'best_model'))

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #Training arguments
    parser.add_argument('--trs', dest='trainset', type=str, default='data/umn/trainset', help='Path to the trainset summary')
    parser.add_argument('--tes', dest='testset', type=str, default='data/umn/testset', help='Path to the testset summary')
    parser.add_argument('--rd', dest='root_dir', type=str, default='/datasets', help='Path to the images')
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.0001, help='Learning rate')
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
