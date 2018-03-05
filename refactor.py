import argparse
import torch
import os
import copy
import torch.nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from tensorboardX import SummaryWriter

import dataset
import utils.plot
import utils.process
import utils.metrics
import models.patch_adversarial_autoencoder

# Train procedure
def train(models, optimizers, datasets, epochs, batch_size, patch_size, z_dim, directory):
    """
    """

    tiny = 1e-15
    alpha = 0.5
    dist = torch.nn.PairwiseDistance(p=2, eps=1e-06)

    phase = ('train', 'test')
    trainset, testset = datasets
    datasets = {'train': trainset, 'test': testset}

    encoder, decoder, discriminator = models
    encoder_optimizer, decoder_optimizer, discriminator_optimizer, regularization_optimizer = optimizers

    best_encoder = copy.deepcopy(encoder)
    best_decoder = copy.deepcopy(decoder)
    best_discriminator = copy.deepcopy(discriminator)
    best_auc = 0

    writer = SummaryWriter(os.path.join(directory, 'logs'))

    for epoch in range(epochs):

        print('Epoch {}'.format(epoch))

        for p in phase:
            if p == 'train':
                encoder.train()
                decoder.train()
                discriminator.train()
            else:
                encoder.eval()
                decoder.eval()
                discriminator.eval()
                labels = []
                reconstruction_errors = []
                discriminator_score = []

            dataloader = DataLoader(datasets[p], batch_size=batch_size, shuffle=True, num_workers=4)

            running_reconstruction_loss = 0
            running_discriminator_loss = 0
            running_regularization_loss = 0
            discriminator_accuracy = 0
            regularization_accuracy = 0

            nb_patch = len(datasets[p]) * ((256 // patch_size)**2)

            for i_batch, sample in enumerate(tqdm(dataloader)):
                #Fetch input data
                inputs = Variable(sample['img'].float().cuda())
                if p == 'test':
                    labels += sample['lbl'].numpy().tolist()

                # Reconstruction phase
                if p == 'train':
                    encoder.zero_grad()
                    decoder.zero_grad()

                z = encoder(inputs)
                reconstruction = decoder(z)
                reconstruction_loss = F.mse_loss(reconstruction, inputs)
                running_reconstruction_loss += reconstruction_loss.data[0]

                if p == 'train':
                    reconstruction_loss.backward()
                    decoder_optimizer.step()
                    encoder_optimizer.step()
                else:
                    r_ = utils.metrics.per_image_error(dist, reconstruction, inputs.view(-1, 3, patch_size, patch_size))
                    reconstruction_errors += r_.data.cpu().numpy().tolist()

                # Discriminator phase
                if p == 'train':
                    encoder.eval()
                    discriminator.zero_grad()

                z_real = Variable(torch.randn(inputs.size(0) * ((256 // patch_size)**2), z_dim) * 5.).cuda()
                z_fake = encoder(inputs)
                # discriminator_real = discriminator(z_real)[1]
                # discriminator_fake = discriminator(z_fake)[1]
                discriminator_real = discriminator(z_real)[0]
                discriminator_fake = discriminator(z_fake)[0]
                discriminator_out = torch.cat((discriminator_real, discriminator_fake), 0)
                zeros = torch.zeros(inputs.size(0) * ((256 // patch_size)**2), 1).float()
                ones = torch.ones(inputs.size(0) * ((256 // patch_size)**2), 1).float()
                groundtruth = Variable(torch.cat((zeros, ones), 0)).cuda()
                # discriminator_loss = -torch.mean(torch.log(discriminator_real + tiny) + torch.log(1 - discriminator_fake + tiny))
                discriminator_loss = torch.nn.functional.binary_cross_entropy_with_logits(discriminator_out, groundtruth)
                running_discriminator_loss += discriminator_loss.data[0]

                discriminator_out = (discriminator_out > 0.5)
                equality = (discriminator_out.float() == groundtruth).float()
                discriminator_accuracy = equality.mean().data[0]

                if p == 'train':
                    discriminator_loss.backward()
                    discriminator_optimizer.step()

                # Regularization phase
                if p == 'train':
                    encoder.train()
                    encoder.zero_grad()
                    discriminator.eval()

                z_fake = encoder(inputs)
                #discriminator_fake = discriminator(z_fake)[1]
                discriminator_fake = discriminator(z_fake)[0]
                zeros = torch.zeros(inputs.size(0) * ((256 // patch_size)**2), 1).float()
                groundtruth = Variable(zeros).cuda()
                regularization_loss = torch.nn.functional.binary_cross_entropy_with_logits(discriminator_fake, groundtruth)
                #regularization_loss = -torch.mean(torch.log(discriminator_fake + tiny))
                running_regularization_loss += regularization_loss.data[0]

                #ones = Variable(torch.ones(inputs.size(0) * ((256 // patch_size)**2), 1).float()).cuda()
                #discriminator_out = (discriminator_fake > 0.5)
                discriminator_out = (discriminator_fake < 0.5)
                equality = (discriminator_out.float() == groundtruth).float()
                #equality = (discriminator_out.float() == ones).float()
                regularization_accuracy = equality.mean().data[0]

                if p == 'train':
                    regularization_loss.backward()
                    regularization_optimizer.step()
                else:
                    discriminator_score += discriminator_fake.data.cpu().numpy().tolist()

            running_reconstruction_loss /= (i_batch + 1)
            running_discriminator_loss /= (i_batch * 2 + 1)
            running_regularization_loss /= (i_batch)

            if p == 'test':
                reconstruction_errors = torch.from_numpy(np.array(reconstruction_errors)).float().cuda()
                discriminator_score = torch.from_numpy(np.array(discriminator_score)).float().cuda()

                image_abnormal_score_alpha_0 = utils.metrics.mean_image_abnormal_score(reconstruction_errors, discriminator_score, 0, patch_size)
                image_abnormal_score_alpha_0 = image_abnormal_score_alpha_0.cpu().numpy().tolist()
                fpr, tpr, thresholds = metrics.roc_curve(labels, image_abnormal_score_alpha_0)
                auc_alpha_0 = metrics.auc(fpr, tpr)

                image_abnormal_score_alpha_05 = utils.metrics.mean_image_abnormal_score(reconstruction_errors, discriminator_score, 0.5, patch_size)
                image_abnormal_score_alpha_05 = image_abnormal_score_alpha_05.cpu().numpy().tolist()
                fpr, tpr, thresholds = metrics.roc_curve(labels, image_abnormal_score_alpha_05)
                auc_alpha_05 = metrics.auc(fpr, tpr)

                image_abnormal_score_alpha_1 = utils.metrics.mean_image_abnormal_score(reconstruction_errors, discriminator_score, 1, patch_size)
                image_abnormal_score_alpha_1 = image_abnormal_score_alpha_1.cpu().numpy().tolist()
                fpr, tpr, thresholds = metrics.roc_curve(labels, image_abnormal_score_alpha_1)
                auc_alpha_1 = metrics.auc(fpr, tpr)
            else:
                auc_alpha_0 = 0
                auc_alpha_05 = 0
                auc_alpha_1 = 0

            print('Reconstruction loss = {}, Discriminator loss = {}, Regularization loss {}, AUC = {}'.format(running_reconstruction_loss, running_discriminator_loss, running_regularization_loss, auc_alpha_05))

            writer.add_scalar('{}/learning_curve/reconstruction_loss'.format(p), running_reconstruction_loss, epoch)
            writer.add_scalar('{}/learning_curve/discriminator_loss'.format(p), running_discriminator_loss, epoch)
            writer.add_scalar('{}/learning_curve/regularization_loss'.format(p), running_regularization_loss, epoch)
            writer.add_scalar('{}/accuracy/discriminator_accuracy'.format(p), discriminator_accuracy, epoch)
            writer.add_scalar('{}/accuracy/regularization_accuracy'.format(p), regularization_accuracy, epoch)
            writer.add_scalar('{}/auc/0'.format(p), auc_alpha_0, epoch)
            writer.add_scalar('{}/auc/05'.format(p), auc_alpha_05, epoch)
            writer.add_scalar('{}/auc/1'.format(p), auc_alpha_1, epoch)

            if p == 'test':
                if auc_alpha_05 > best_auc:
                    best_auc = auc_alpha_05
                    best_encoder = copy.deepcopy(encoder)
                    best_decoder = copy.deepcopy(decoder)
                    best_discriminator = copy.deepcopy(discriminator)
                if epoch % 10 == 0:
                    #Save model
                    torch.save(encoder.state_dict(), os.path.join(directory, 'serial', 'encoder_{}'.format(epoch)))
                    torch.save(decoder.state_dict(), os.path.join(directory, 'serial', 'decoder_{}'.format(epoch)))
                    torch.save(discriminator.state_dict(), os.path.join(directory, 'serial', 'discriminator_{}'.format(epoch)))

                    #Plot example of reconstructed images
                    reconstruction = reconstruction.view(-1, 3, 256, 256)
                    pred = utils.process.deprocess(reconstruction)
                    pred = pred.data.cpu().numpy()
                    pred = np.rollaxis(pred, 1, 4)
                    inputs = utils.process.deprocess(inputs)
                    inputs = inputs.data.cpu().numpy()
                    inputs = np.rollaxis(inputs, 1, 4)
                    utils.plot.plot_reconstruction_images(inputs, pred, os.path.join(directory, 'example_reconstruction', 'epoch_{}.svg'.format(epoch)))

    writer.export_scalars_to_json(os.path.join(directory, 'logs', 'scalars.json'))
    writer.close()

    return best_encoder, best_decoder, best_discriminator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bs', type=int, default=100, help='Batch size')
    parser.add_argument('-e', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lrr', type=float, default=0.001, help='Reconstruction learning rate')
    parser.add_argument('--lra', type=float, default=0.0005, help='Adversarial learning rate')
    parser.add_argument('--dir', type=str, default='adversarial_patch_autoencoder', help='Directory to store results')
    parser.add_argument('-p', type=int, default=32, help='Patch size')
    parser.add_argument('--trs', type=str, default='data/umn_normal_trainset', help='Path to the trainset summary')
    parser.add_argument('--tes', type=str, default='data/umn_testset', help='Path to the testset summary')
    parser.add_argument('--rd', type=str, default='/home/scom/data/umn', help='Path to the images')

    parser.add_argument('-f', type=int, default=8, help='Number of convolutional filters')
    parser.add_argument('-b', type=int, default=1, help='Number of convolutional blocks')
    parser.add_argument('-l', type=int, default=1, help='Number of convolutional layers')
    parser.add_argument('-z', type=int, default=2, help='Latent size')
    parser.add_argument('-d', type=str, default='1000,1000', help='Discriminator\'s hidden architecture')

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    if not os.path.exists(os.path.join(args.dir, 'serial')):
        os.makedirs(os.path.join(args.dir, 'serial'))
    if not os.path.exists(os.path.join(args.dir, 'plots')):
        os.makedirs(os.path.join(args.dir, 'plots'))
    if not os.path.exists(os.path.join(args.dir, 'example_reconstruction')):
        os.makedirs(os.path.join(args.dir, 'example_reconstruction'))
    if not os.path.exists(os.path.join(args.dir, 'logs')):
        os.makedirs(os.path.join(args.dir, 'logs'))

    #Datasets
    trainset = dataset.VideoDataset(args.trs, args.rd)
    testset = dataset.VideoDataset(args.tes, args.rd)
    datasets = [trainset, testset]

    #Models
    encoder = models.patch_adversarial_autoencoder.Encoder(args.f, args.l, args.b, args.z, args.p)
    encoder = encoder.cuda()
    decoder = models.patch_adversarial_autoencoder.Decoder(encoder.last_map_dim, args.l, args.b, args.z)
    decoder = decoder.cuda()
    discriminator = models.patch_adversarial_autoencoder.Discriminator(args.z, args.d)
    discriminator = discriminator.cuda()
    print(encoder)
    print(decoder)
    print(discriminator)
    models = [encoder, decoder, discriminator]

    #Optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lrr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lrr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.lra)
    regularization_optimizer = optim.Adam(encoder.parameters(), lr=args.lra)
    optimizers = [encoder_optimizer, decoder_optimizer, discriminator_optimizer, regularization_optimizer]

    encoder, decoder, discriminator = train(models, optimizers, datasets, args.e, args.bs, args.p, args.z, args.dir)

    torch.save(encoder.state_dict(), os.path.join(args.dir, 'serial', 'best_encoder'))
    torch.save(decoder.state_dict(), os.path.join(args.dir, 'serial', 'best_decoder'))
    torch.save(discriminator.state_dict(), os.path.join(args.dir, 'serial', 'best_discriminator'))
