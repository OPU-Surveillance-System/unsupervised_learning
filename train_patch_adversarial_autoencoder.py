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
import models.patch_adversarial_autoencoder
import utils.metrics
import utils.plot
import utils.process

def train(networks, loss_functions, optimizers, trainset, testset, epoch, batch_size, latent_size, patch_size, directory):
    """
    Train an adversarial autoencoder and log the process
    Args:
        networks (list of torch.nn.Module): List of models to train (encoder, decoder, discriminator)
        loss_functions (list of torch.optim.Module): Loss functions (reconstruction, adversarial)
        optimizers (list of torch.optim.Optimizer): Optimizers (encoder, decoder, discriminator, adversarial encoder)
        trainset (torch.utils.data.Dataset): Training set
        testset (torch.utils.data.Dataset): Test set
        epoch (int): Number of training epochs
        batch_size (int): Mini batch size
        latent_size (int):
        patch_size (int):
        directory (str): Directory to store the logs
    """

    encoder, decoder, discriminator = networks
    reconstruction_loss_function, adversarial_loss_function = loss_functions
    encoder_optimizer, decoder_optimizer, discriminator_optimizer, adversarial_encoder_optimizer = optimizers
    phase = ('train', 'test')
    datasets = {'train': trainset, 'test': testset}
    dist = torch.nn.PairwiseDistance(p=2, eps=1e-06)
    writer = SummaryWriter(os.path.join(directory, 'logs'))

    for e in range(epoch):
        print('Epoch {}'.format(e))
        for p in phase:
            running_reconstruction_loss = 0
            running_discriminator_loss = 0
            running_adversarial_loss = 0
            dataloader = DataLoader(datasets[p], batch_size=batch_size, shuffle=True, num_workers=4)

            nb_patch = len(datasets[p]) * ((256//patch_size)**2)
            nb_discriminator_sample = nb_patch * 2

            for i_batch, sample in enumerate(tqdm(dataloader)):

                #Reconstruction
                if p == 'train':
                    encoder.train()
                    decoder.train()
                    encoder.zero_grad()
                    decoder.zero_grad()
                else:
                    encoder.eval()
                    decoder.eval()
                inputs = Variable(sample['img'].float().cuda())
                latent = encoder(inputs)
                reconstruction = decoder(latent)
                loss = reconstruction_loss_function(reconstruction, inputs.view(-1, 3, patch_size, patch_zise))
                if p == 'train':
                    loss.backward()
                    decoder.step()
                    encoder.step()
                running_reconstruction_loss += loss.data[0]

                #Discriminator
                encoder.eval()
                if p == 'train':
                    discriminator.train()
                    discriminator.zero_grad()
                else:
                    discriminator.eval()
                z_real = Variable(torch.randn(inputs.size(0) * ((256//patch_size)**2), latent_size).cuda()) #Sample from N(0, 1)
                z_fake = encoder(inputs)
                logits_real = discriminator(z_real)
                logits_fake = discriminator(z_fake)
                labels = Variable(torch.cat((torch.zeros(logits_real.size(0)), torch.ones(logits_fake.size(0))), 0).float().cuda())
                loss = adversarial_loss_function(torch.cat((logits_real, logits_fake), 0), labels)
                if p == 'train':
                    loss.backward()
                    discriminator_optimizer.step()
                running_discriminator_loss += loss.data[0]

                #Adversarial
                discriminator.eval()
                if p == 'train':
                    encoder.train()
                    encoder.zero_grad()
                else:
                    encoder.eval()
                z_real = encoder(inputs)
                logits_real = discriminator(z_real)
                labels = Variable(torch.zeros(logits_real.size(0)).float().cuda())
                loss = adversarial_loss_function(logits_real, labels)
                if p == 'train':
                    loss.backward()
                    adversarial_encoder_optimizer.step()
                running_adversarial_loss += loss.data[0]

            #Computes epoch average losses
            epoch_reconstruction_loss = running_reconstruction_loss / nb_patch
            epoch_discriminator_loss = running_discriminator_loss / nb_discriminator_sample
            epoch_adversarial_loss = running_adversarial_loss / nb_patch
            writer.add_scalar('learning_curve/reconstruction_loss/{}'.format(p), epoch_reconstruction_loss, e)
            writer.add_scalar('learning_curve/discriminator_loss/{}'.format(p), epoch_discriminator_loss, e)
            writer.add_scalar('learning_curve/adversarial_loss/{}'.format(p), epoch_adversarial_loss, e)

            if p == 'test':
                if e % 10 == 0:
                    #Save model
                    torch.save(encoder.state_dict(), os.path.join(directory, 'serial', 'encoder_{}'.format(e)))
                    torch.save(decoder.state_dict(), os.path.join(directory, 'serial', 'decoder_{}'.format(e)))
                    torch.save(discriminator.state_dict(), os.path.join(directory, 'serial', 'discriminator_{}'.format(e)))

                    #Plot example of reconstructed images
                    reconstruction = reconstruction.view(-1, 3, 256, 256)
                    pred = utils.process.deprocess(reconstruction)
                    pred = pred.data.cpu().numpy()
                    pred = np.rollaxis(pred, 1, 4)
                    inputs = utils.process.deprocess(inputs)
                    inputs = inputs.data.cpu().numpy()
                    inputs = np.rollaxis(inputs, 1, 4)
                    utils.plot.plot_reconstruction_images(inputs, pred, os.path.join(directory, 'example_reconstruction', 'epoch_{}.svg'.format(e)))
    writer.export_scalars_to_json(os.path.join(directory, 'logs', 'scalars.json'))
    writer.close()

    return encoder, decoder, discriminator

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

    #Networks
    encoder = models.patch_adversarial_autoencoder.Encoder(args.nb_f, args.nb_l, args.nb_b, args.latent_size, args.patch)
    decoder = models.patch_adversarial_autoencoder.Decoder(encoder.last_map_dim, args.nb_l, args.nb_b, args.latent_size)
    discriminator = models.patch_adversarial_autoencoder.Discriminator(args.latent_size, args.denses)
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    discriminator = discriminator.cuda()
    print(encoder)
    print(decoder)
    print(discriminator)
    networks = [encoder, decoder, discriminator]

    #Loss functions
    reconstruction_loss_function = torch.nn.MSELoss()
    adversarial_loss_function = torch.nn.BCEWithLogitsLoss()
    loss_functions = [reconstruction_loss_function, adversarial_loss_function]

    #Optimizers
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), args.learning_rate_reconstruction)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), args.learning_rate_reconstruction)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), args.learning_rate_adversarial)
    adversarial_encoder_optimizer = torch.optim.Adam(encoder.parameters(), args.learning_rate_adversarial)
    optimizers = [encoder_optimizer, decoder_optimizer, discriminator_optimizer, adversarial_encoder_optimizer]


    #Datasets
    trainset = dataset.VideoDataset(args.trainset, args.root_dir)
    testset = dataset.VideoDataset(args.testset, args.root_dir)

    #Train the model and save it
    encoder, decoder, discriminator = train(networks, loss_functions, optimizers, trainset, testset, args.epoch, args.batch_size, args.latent_size, args.patch, args.directory)
    #torch.save(best_model.state_dict(), os.path.join(args.directory, 'serial', 'best_model'))

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #Training arguments
    parser.add_argument('--trs', dest='trainset', type=str, default='data/umn_normal_trainset', help='Path to the trainset summary')
    parser.add_argument('--tes', dest='testset', type=str, default='data/umn_testset', help='Path to the testset summary')
    parser.add_argument('--rd', dest='root_dir', type=str, default='/datasets', help='Path to the images')
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--lrr', dest='learning_rate_reconstruction', type=float, default=0.001, help='Reconstruction learning rate')
    parser.add_argument('--lra', dest='learning_rate_adversarial', type=float, default=0.0005, help='Adversarial learning rate')
    parser.add_argument('--ep', dest='epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--dir', dest='directory', type=str, default='train_autoencoder', help='Directory to store results')
    #Model arguments
    parser.add_argument('-f', dest='nb_f', type=int, default=16, help='Number of filters in the first downsampling block')
    parser.add_argument('-l', dest='nb_l', type=int, default=1, help='Number of convolutinal layers per block')
    parser.add_argument('-b', dest='nb_b', type=int, default=1, help='Number of upsampling blocks')
    parser.add_argument('--ls', dest='latent_size', type=int, default=512, help='Latent size')
    parser.add_argument('-d', dest='denses', type=str, default='1024, 1024', help='Discriminator hidden layers')
    parser.add_argument('-i', dest='ips', type=int, default=256, help='Image height (assume width = height)')
    parser.add_argument('-p', dest='patch', type=int, default=32, help='Image patch size')
    args = parser.parse_args()

    main(args)
