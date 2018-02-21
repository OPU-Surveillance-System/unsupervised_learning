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
import models.adversarial_autoencoder
import utils.metrics
import utils.plot
import utils.process

def train(encoder, decoder, discriminator, reconstruction_loss_function, adversarial_loss_function, encoder_optimizer, decoder_optimizer, discriminator_optimizer, adversarial_encoder_optimizer, trainset, testset, epoch, batch_size, latent_size, directory):
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
    zeros = torch.zeros(batch_size) * -1
    ones = torch.ones(batch_size)
    discriminator_labels = Variable(torch.cat((zeros, ones), 0).float().cuda())
    zeros = Variable(zeros.float().cuda())

    for e in range(epoch):
        print('Epoch {}'.format(e))
        for p in phase:
            reconstruction_loss = 0
            discriminator_loss = 0
            adversarial_loss = 0

            dataloader = DataLoader(datasets[p], batch_size=batch_size, shuffle=True, num_workers=4)
            for i_batch, sample in enumerate(tqdm(dataloader)):
                model.zero_grad()
                inputs = Variable(sample['img'].float().cuda())

                #Reconstruction training
                if p == 'train':
                    encoder.train()
                    decoder.train()
                else:
                    encoder.eval()
                    decoder.eval()
                z = encoder(inputs)
                logits, pred = decoder(z)
                loss = reconstruction_loss_function(logits, inputs)
                if p == 'train':
                    loss.backward()
                    decoder_optimizer.step()
                    encoder_optimizer.step()
                reconstruction_loss += loss.data[0]

                #Discriminator training
                encoder.eval()
                if p == 'train':
                    discriminator.train()
                else:
                    discriminator.eval()
                z_real = Variable(torch.randn(batch_size, latent_size)).cuda() #~N(0, 1)
                z_fake = z
                real_logits, real_pred = discriminator(z_real)
                fake_logits, fake_pred = discriminator(z_fake)
                logits = torch.cat((real_logits, fake_logits), 0)
                loss = adversarial_loss_function(logits, discriminator_labels)
                if p == 'train':
                    loss.backward()
                    discriminator_optimizer.step()
                discriminator_loss += loss.data[0]

                #Generator training
                discriminator.eval()
                if p == 'train':
                    encoder.train()
                else:
                    encoder.eval()
                logits, pred = discriminator(z_fake)
                loss = adversarial_loss_function(logits, zeros)
                if p == 'train':
                    loss.backward()
                    adversarial_encoder_optimizer.step()
                adversarial_loss += loss.data[0]

            reconstruction_loss /= len(datasets[p])
            discriminator_loss /= (2 * batch_size * (i_batch + 1))
            adversarial_loss /= (batch_size * (i_batch + 1))
            writer.add_scalar('learning_curve/reconstruction_loss/{}'.format(p), reconstruction_loss, e)
            writer.add_scalar('learning_curve/discriminator_loss/{}'.format(p), discriminator_loss, e)
            writer.add_scalar('learning_curve/adversarial_loss/{}'.format(p), adversarial_loss, e)

            print('{} -- Reconstruction Loss: {}, Discriminator Loss: {}, Adversarial Loss: {}'.format(p, reconstruction_loss, discriminator_loss, adversarial_loss))
            # if p == 'test':
            #     writer.add_scalar('auc', auc, e)
            #     if auc > best_auc:
            #         best_auc = auc
            #         best_model = copy.deepcopy(model)
            #     if e % 10 == 0:
            #         #Save model
            #         torch.save(model.state_dict(), os.path.join(directory, 'serial', 'model_{}'.format(e)))
            #
            #         #Plot example of reconstructed images
            #         pred = utils.process.deprocess(pred)
            #         pred = pred.data.cpu().numpy()
            #         pred = np.rollaxis(pred, 1, 4)
            #         inputs = utils.process.deprocess(inputs)
            #         inputs = inputs.data.cpu().numpy()
            #         inputs = np.rollaxis(inputs, 1, 4)
            #         utils.plot.plot_reconstruction_images(inputs, pred, os.path.join(directory, 'example_reconstruction', 'epoch_{}.svg'.format(e)))
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
    if not os.path.exists(os.path.join(args.directory, 'example_reconstruction')):
        os.makedirs(os.path.join(args.directory, 'example_reconstruction'))
    if not os.path.exists(os.path.join(args.directory, 'logs')):
        os.makedirs(os.path.join(args.directory, 'logs'))

    #Write arguments in a file
    d = vars(args)
    with open(os.path.join(args.directory, 'hyper-parameters'), 'w') as f:
        for k in d.keys():
            f.write('{}:{}\n'.format(k, d[k]))

    #Models
    encoder = models.adversarial_autoencoder.Encoder(args.nb_f, args.nb_l, args.nb_b, args.latent_size, args.ips)
    decoder = models.adversarial_autoencoder.Decoder(encoder.out_dim, args.nb_l, args.nb_b, args.latent_size)
    discriminator = models.adversarial_autoencoder.Discriminator(args.latent_size, args.dense)
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    discriminator = discriminator.cuda()
    print(encoder)
    print(decoder)
    print(discriminator)

    #Loss functions
    reconstruction_loss_function = torch.nn.MSELoss()
    adversarial_loss_function = torch.nn.BCEWithLogitsLoss()

    #Optimizers
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), args.reconstruction_learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), args.reconstruction_learning_rate)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), args.adversarial_learning_rate)
    adversarial_encoder_optimizer = torch.optim.Adam(encoder.parameters(), args.adversarial_learning_rate)

    #Datasets
    trainset = dataset.VideoDataset(args.trainset, args.root_dir)
    testset = dataset.VideoDataset(args.testset, args.root_dir)

    #Train the models and save them
    best_encoder, best_decoder, best_discriminator = train(encoder, decoder, discriminator, reconstruction_loss_function, adversarial_loss_function, encoder_optimizer, decoder_optimizer, discriminator_optimizer, adversarial_encoder_optimizer, trainset, testset, args.epoch, args.batch_size, args.latent_size, args.directory)
    torch.save(best_encoder.state_dict(), os.path.join(args.directory, 'serial', 'best_encoder'))
    torch.save(best_decoder.state_dict(), os.path.join(args.directory, 'serial', 'best_decoder'))
    torch.save(best_discriminator.state_dict(), os.path.join(args.directory, 'serial', 'best_discriminator'))
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #Training arguments
    parser.add_argument('--trs', dest='trainset', type=str, default='data/umn_normal_trainset', help='Path to the trainset summary')
    parser.add_argument('--tes', dest='testset', type=str, default='data/umn_testset', help='Path to the testset summary')
    parser.add_argument('--rd', dest='root_dir', type=str, default='/datasets', help='Path to the images')
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--rlr', dest='reconstruction_learning_rate', type=float, default=0.0001, help='Reconstruction loss learning rate')
    parser.add_argument('--alr', dest='adversarial_learning_rate', type=float, default=0.0001, help='Adversarial loss learning rate')
    parser.add_argument('--ep', dest='epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
    #Model arguments
    parser.add_argument('-f', dest='nb_f', type=int, default=32, help='Number of filters in the first downsampling block')
    parser.add_argument('-l', dest='nb_l', type=int, default=1, help='Number of convolutinal layers per block')
    parser.add_argument('-b', dest='nb_b', type=int, default=2, help='Number of upsampling blocks')
    parser.add_argument('-d', dest='dense', type=str, default='512,256', help='Discriminator layers (eg. 512,256)')
    parser.add_argument('-i', dest='ips', type=int, default=256, help='Image height (assume width = height)')
    parser.add_argument('-s', dest='latent_size', type=int, default=512, help='Latent dimension')
    args = parser.parse_args()

    main(args)
