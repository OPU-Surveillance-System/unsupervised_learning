import torch
import math
from torch.autograd import Variable

class Encoder(torch.nn.Module):
    def __init__(self, nb_f, nb_l, nb_b, latent_size, patch=32):
        super(Encoder, self).__init__()
        self.nb_f = nb_f
        self.nb_l = nb_l
        self.nb_b = nb_b
        self.latent_size = latent_size
        self.patch = patch

        def downsampling_block(in_dim, nb_f, nb_l):
            layers = []
            for n in range(nb_l):
                layers.append(torch.nn.Conv2d(in_dim, nb_f, (3, 3), padding=1))
                layers.append(torch.nn.Dropout2d(p=0.25))
                layers.append(torch.nn.ReLU())
                in_dim = nb_f
            layers.append(torch.nn.MaxPool2d((2, 2), (2, 2)))

            return layers

        #Conv part
        layers = []
        prev_in = 3 #RGB images
        prev_f = self.nb_f
        for n in range(self.nb_b):
            layers += downsampling_block(prev_in, prev_f, self.nb_l)
            prev_in = prev_f
            prev_f *= 2
        self.conv = torch.nn.Sequential(*layers)
        self.last_map_dim = ','.join([str(prev_f//2), str(self.patch//(2**self.nb_b)), str(self.patch//(2**self.nb_b))])

        #Bottleneck
        flatten = ((self.patch//(2**self.nb_b))**2)*(prev_f//2) #last_feature_map.h * last_feature_map.w * last_feature_map.c
        self.mu = torch.nn.Linear(flatten, self.latent_size)
        self.sigma = torch.nn.Linear(flatten, self.latent_size)

    def forward(self, x):
        #Encode
        x = x.view(-1, 3, self.patch, self.patch)
        x = self.conv(x)
        x = x.view(x.size(0), -1) #Flatten x
        mu = self.mu(x)
        sigma = self.sigma(x)

        return mu, sigma

################################################################################

class Decoder(torch.nn.Module):
    def __init__(self, encoder_dim, nb_l, nb_b, latent_size):
        super(Decoder, self).__init__()
        self.encoder_dim = [int(d) for d in encoder_dim.split(',')]
        self.nb_l = nb_l
        self.nb_b = nb_b
        self.latent_size = latent_size

        def upsampling_block(in_dim, nb_f, nb_l):
            layers = [torch.nn.Upsample(scale_factor=2, mode='bilinear')]
            for n in range(nb_l):
                layers.append(torch.nn.Conv2d(in_dim, nb_f, (3, 3), padding=1))
                layers.append(torch.nn.Dropout2d(p=0.25))
                layers.append(torch.nn.ReLU())
                in_dim = nb_f

            return layers

        #Bottleneck
        out = self.encoder_dim[0] * self.encoder_dim[1] * self.encoder_dim[2]
        self.bottleneck = torch.nn.Linear(self.latent_size, out)

        #Conv part
        layers = []
        prev_f = self.encoder_dim[0]
        next_f = prev_f // 2
        for n in range(self.nb_b):
            layers += upsampling_block(prev_f, next_f, self.nb_l)
            prev_f = next_f
            next_f //= 2
        layers.append(torch.nn.Conv2d(prev_f, 3, (3, 3), padding=1))
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, z):
        x = self.bottleneck(z)
        x = x.view(-1, self.encoder_dim[0], self.encoder_dim[1], self.encoder_dim[2])
        x = self.conv(x)

        return x
