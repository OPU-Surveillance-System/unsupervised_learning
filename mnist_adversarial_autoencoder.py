import argparse
import torch
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
import os

import utils.plot

# Training settings
parser = argparse.ArgumentParser(description='PyTorch semi-supervised MNIST')

parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')

args = parser.parse_args()

z_dim = 2
X_dim = 784
y_dim = 10
batch_size = args.batch_size
N = 1000
epochs = args.epochs
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.lin1 = nn.Linear(X_dim, N)
        # self.lin2 = nn.Linear(N, N)
        # self.lin3gauss = nn.Linear(N, z_dim)
        layers = []
        layers.append(nn.Conv2d(1, 8, (3, 3), (1, 1), padding=1))
        layers.append(nn.Dropout2d(p=0.2))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d())
        self.conv = nn.Sequential(*layers)
        self.latent = nn.Linear(8*((28//2)**2), z_dim)

    def forward(self, x):
        # x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        # x = F.relu(x)
        # x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        # x = F.relu(x)
        # xgauss = self.lin3gauss(x)
        #
        # return xgauss

        x = self.conv(x)
        x = x.view((batch_size, -1))
        x = self.latent(x)

        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.lin1 = nn.Linear(z_dim, N)
        # self.lin2 = nn.Linear(N, N)
        # self.lin3 = nn.Linear(N, X_dim)
        self.latent = nn.Linear(z_dim, 8*((28//2)**2))
        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        layers.append(nn.Conv2d(8, 1, (3, 3), (1, 1), padding=1))
        layers.append(nn.Dropout2d(p=0.2))
        layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # x = self.lin1(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = F.relu(x)
        # x = self.lin2(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.lin3(x)
        #
        # return x

        x = self.latent(x)
        x = x.view((-1, 8, 14, 14))
        x = self.conv(x)

        return x

#Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)

        return F.sigmoid(self.lin3(x))

# Utility functions
def report_loss(epoch, reconstruction_loss, discriminator_loss, generator_loss):
    '''
    Print loss
    '''
    print('Epoch {}: discriminator loss={:.4}, generator loss={:.4}, reconstruction loss={:.4}'.format(epoch, discriminator_loss.data[0],
                                                                                   generator_loss.data[0],
                                                                                   reconstruction_loss.data[0]))

# Train procedure
def train(encoder, decoder, discriminator, encoder_optimizer, decoder_optimizer, discriminator_optimizer, generator_optimizer):
    '''
    Train procedure for one epoch.
    '''

    TINY = 1e-15
    # Set the networks in train mode (apply dropout when needed)
    encoder.train()
    decoder.train()
    discriminator.train()

    for i in range(mnist.train.num_examples//batch_size):

        # Load batch and normalize samples to be in [-1, 1]
        img = mnist.train.next_batch(batch_size)[0]
        img = (img - 0.5) / 0.5
        img = img.reshape((batch_size, 1, 28, 28))
        img = Variable(torch.from_numpy(img)).cuda()

        # Init gradients
        encoder.zero_grad()
        decoder.zero_grad()
        discriminator.zero_grad()

        # Reconstruction phase
        z_sample = encoder(img)
        reconstruction = decoder(z_sample)
        reconstruction_loss = F.mse_loss(reconstruction, img)

        reconstruction_loss.backward()
        decoder_optimizer.step()
        encoder_optimizer.step()

        # Discriminator phase
        encoder.eval()

        z_real = Variable(torch.randn(batch_size, z_dim) * 5.).cuda()
        z_fake = encoder(img)
        discriminator_real = discriminator(z_real)
        discriminator_fake = discriminator(z_fake)
        discriminator_loss = -torch.mean(torch.log(discriminator_real + TINY) + torch.log(1 - discriminator_fake + TINY))

        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Generator
        encoder.zero_grad()
        encoder.train()
        discriminator.eval()

        z_fake = encoder(img)
        discriminator_fake = discriminator(z_fake)
        generator_loss = -torch.mean(torch.log(discriminator_fake + TINY))

        generator_loss.backward()
        generator_optimizer.step()

        encoder.zero_grad()
        decoder.zero_grad()
        discriminator.zero_grad()

    return reconstruction_loss, discriminator_loss, generator_loss

def generate_model():
    #Models
    encoder = Encoder().cuda()
    decoder = Decoder().cuda()
    discriminator = Discriminator().cuda()

    #learning rates
    reconstruction_learning_rate = 0.0001
    regularization_learning_rate = 0.00005

    # Set optimizators
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=reconstruction_learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=reconstruction_learning_rate)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=regularization_learning_rate)
    generator_optimizer = optim.Adam(encoder.parameters(), lr=regularization_learning_rate)

    for epoch in range(epochs):
        reconstruction_loss, discriminator_loss, generator_loss = train(encoder, decoder, discriminator, encoder_optimizer, decoder_optimizer, discriminator_optimizer, generator_optimizer)
        if epoch % 10 == 0:
            report_loss(epoch, reconstruction_loss, discriminator_loss, generator_loss)
            z = Variable(torch.randn(4, z_dim) * 5.).cuda()
            output = decoder(z)
            output = (output + 1) * 0.5
            output = output.data.cpu().numpy()
            output = np.reshape(output, (4, 28, 28))
            utils.plot.plot_generated_images(output, os.path.join('mnist', 'generated_{}'.format(epoch)))


    return encoder, decoder, discriminator

if __name__ == '__main__':
    if not os.path.exists('mnist'):
        os.makedirs('mnist')
    encoder, decoder, discriminator = generate_model()
