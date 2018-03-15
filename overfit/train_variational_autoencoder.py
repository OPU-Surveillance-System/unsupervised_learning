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
import overfit.variational_autoencoder
import utils.metrics
import utils.plot
import utils.process

def sample_z(mu, sigma):
    # Using reparameterization trick to sample from a gaussian
    eps = Variable(torch.randn(mu.size(0), mu.size(1))).float().cuda()
    z = mu + torch.exp(sigma / 2) * eps

    return z

def train(models, optimizers, trainset, testset, epoch, batch_size, reg, directory):
    """
    Train a model and log the process
    Args:
        model (torch.nn.Module): Model to train (autoencoder)
        optimizer (torch.optim.Optimizer): Optimizer
        trainset (torch.utils.data.Dataset): Training set
        testset (torch.utils.data.Dataset): Test set
        epoch (int): Number of training epochs
        batch_size (int): Mini batch size
        directory (str): Directory to store the logs
    """

    encoder, decoder = models
    en_optimizer, de_optimizer = optimizers

    phase = ('train', 'test')
    datasets = {'train': trainset, 'test': testset}
    dist = torch.nn.PairwiseDistance(p=2, eps=1e-06)
    best_auc = 0
    best_encoder = copy.deepcopy(encoder)
    best_decoder = copy.deepcopy(decoder)
    writer = SummaryWriter(os.path.join(directory, 'logs'))

    for e in range(epoch):
        print('Epoch {}'.format(e))
        for p in phase:
            if p == 'train':
                encoder.train()
                decoder.train()
            else:
                labels = []
                errors = []
                encoder.eval()
                decoder.eval()
            running_reconstruction_loss = 0
            running_regularization_loss = 0
            dataloader = DataLoader(datasets[p], batch_size=batch_size, shuffle=True, num_workers=4)
            for i_batch, sample in enumerate(tqdm(dataloader)):
                encoder.zero_grad()
                decoder.zero_grad()
                inputs = Variable(sample['img'].float().cuda())
                nb_channel = inputs.size(1)
                mu, sigma = encoder(inputs)
                z = sample_z(mu, sigma)
                logits = decoder(z)
                reconstruction_loss = torch.nn.functional.mse_loss(logits, inputs, size_average=True)
                regularization_loss = 0.5 * torch.mean((mu**2) + (torch.exp(sigma)**2) - torch.log(torch.exp(sigma)) - 1)
                #regularization_loss = 0.5 * torch.sum(torch.exp(sigma) + mu**2 - 1. - sigma)
                loss = reconstruction_loss + reg * regularization_loss
                if p == 'train':
                    loss.backward()
                    en_optimizer.step()
                    de_optimizer.step()
                running_reconstruction_loss += reconstruction_loss.data[0]
                running_regularization_loss += regularization_loss.data[0]
                if p == 'test':
                    tmp = utils.metrics.per_image_error(dist, logits, inputs)
                    errors += tmp.data.cpu().numpy().tolist()
                    labels += sample['lbl'].numpy().tolist()

            if p == 'test':
                errors = torch.from_numpy(np.array(errors)).cuda()
                utils.metrics.normalize_reconstruction_errors(errors)
                errors = errors.cpu().numpy()
                fpr, tpr, thresholds = metrics.roc_curve(labels, errors)
                auc = metrics.auc(fpr, tpr)
            else:
                auc = 0
            epoch_reconstruction_loss = running_reconstruction_loss / len(datasets[p])
            epoch_regularization_loss = running_regularization_loss / len(datasets[p])
            writer.add_scalar('{}/learning_curve/reconstruction_loss'.format(p), epoch_reconstruction_loss, e)
            writer.add_scalar('{}/learning_curve/regularization_loss'.format(p), epoch_regularization_loss, e)
            print('{} -- Reconstruction loss: {}, Regularization loss: {}, AUC: {}'.format(p, epoch_reconstruction_loss, epoch_regularization_loss, auc))
            if p == 'test':
                writer.add_scalar('auc', auc, e)
                if auc > best_auc:
                    best_auc = auc
                    best_encoder = copy.deepcopy(encoder)
                    best_decoder = copy.deepcopy(decoder)
                # if e % 10 == 0:
                #     #Save model
                #     torch.save(encoder.state_dict(), os.path.join(directory, 'serial', 'encoder_{}'.format(e)))
                #     torch.save(decoder.state_dict(), os.path.join(directory, 'serial', 'decoder_{}'.format(e)))
            if e % 10 == 0:
                #Plot example of reconstructed images
                pred = utils.process.deprocess(logits)
                #pred = torch.nn.functional.sigmoid(logits)
                pred = pred.data.cpu().numpy()
                pred = np.rollaxis(pred, 1, 4)
                inputs = utils.process.deprocess(inputs)
                inputs = inputs.data.cpu().numpy()
                inputs = np.rollaxis(inputs, 1, 4)
                utils.plot.plot_reconstruction_images(inputs, pred, os.path.join(directory, 'example_reconstruction_{}'.format(p), 'epoch_{}.svg'.format(e)))
    writer.export_scalars_to_json(os.path.join(directory, 'logs', 'scalars.json'))
    writer.close()

    return best_encoder, best_decoder

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
    if args.mode == 'L':
        input_dim = 1
    else:
        input_dim = 3

    encoder = overfit.variational_autoencoder.Encoder(in_dim=input_dim,
                                                           nb_f=args.nb_f,
                                                           nb_l=args.nb_l,
                                                           nb_b=args.nb_b,
                                                           latent_size=args.latent_size)
    decoder = overfit.variational_autoencoder.Decoder(out_dim=input_dim,
                                                           encoder_dim=encoder.last_map_dim,
                                                           nb_l=args.nb_l,
                                                           nb_b=args.nb_b,
                                                           latent_size=args.latent_size)
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    model = (encoder, decoder)
    print(encoder)
    print(decoder)

    en_optimizer = torch.optim.Adam(encoder.parameters(), args.learning_rate)
    de_optimizer = torch.optim.Adam(decoder.parameters(), args.learning_rate)
    optimizers = (en_optimizer, de_optimizer)

    trainset = dataset.VideoDataset(args.trainset, args.root_dir, args.mode)
    testset = dataset.VideoDataset(args.testset, args.root_dir, args.mode)

    #Train the model and save it
    best_encoder, best_decoder = train(model, optimizers, trainset, testset, args.epoch, args.batch_size, args.regularization, args.directory)
    # torch.save(best_encoder.state_dict(), os.path.join(args.directory, 'serial', 'best_encoder'))
    # torch.save(best_decoder.state_dict(), os.path.join(args.directory, 'serial', 'best_decoder'))

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #Training arguments
    parser.add_argument('--trs', dest='trainset', type=str, default='data/umn_normal_trainset', help='Path to the trainset summary')
    parser.add_argument('--tes', dest='testset', type=str, default='data/umn_testset', help='Path to the testset summary')
    parser.add_argument('--rd', dest='root_dir', type=str, default='/datasets', help='Path to the images')
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--reg', dest='regularization', type=float, default=0.0001, help='Regularization')
    parser.add_argument('--ep', dest='epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
    parser.add_argument('-m', dest='mode', type=str, default='RGB', help='Image mode (L: black and white, RGB: RGB)')
    #Model arguments
    parser.add_argument('-f', dest='nb_f', type=int, default=16, help='Number of filters in the first downsampling block')
    parser.add_argument('-l', dest='nb_l', type=int, default=1, help='Number of convolutinal layers per block')
    parser.add_argument('-b', dest='nb_b', type=int, default=2, help='Number of upsampling blocks')
    parser.add_argument('-z', dest='latent_size', type=int, default=512, help='Size of latent codes')
    parser.add_argument('-i', dest='ips', type=int, default=256, help='Image height (assume width = height)')
    args = parser.parse_args()

    main(args)
