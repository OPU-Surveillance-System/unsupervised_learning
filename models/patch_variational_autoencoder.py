import torch
import math

class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, nb_f, nb_l, nb_b, fc=None, input_size=256, patch=32):
        super(Autoencoder, self).__init__()
        self.in_dim = 3
        self.nb_f = nb_f
        self.nb_l = nb_l
        self.nb_b = nb_b
        self.fc = fc
        self.ips = input_size
        self.patch = patch
        self.reshape = self.patch//(2**self.nb_b)

        def downsampling_block(in_dim, nb_f, nb_l):
            layers = []
            for n in range(nb_l):
                layers.append(torch.nn.Conv2d(in_dim, nb_f, (3, 3), padding=1))
                layers.append(torch.nn.SELU())
                in_dim = nb_f
            layers.append(torch.nn.MaxPool2d((2, 2), (2, 2)))

            return layers

        def upsampling_block(in_dim, nb_f, nb_l):
            #layers = [torch.nn.ConvTranspose2d(in_dim, nb_f, (2, 2), (2, 2))]
            layers = [torch.nn.Upsample(scale_factor=2, mode='bilinear')]
            layers.append(torch.nn.SELU())
            for n in range(nb_l):
                layers.append(torch.nn.Conv2d(in_dim, nb_f, (3, 3), padding=1))
                layers.append(torch.nn.SELU())
                in_dim = nb_f

            return layers

        #Encoder
        layers = []
        prev_in = self.in_dim
        prev_f = self.nb_f
        for n in range(self.nb_b):
            layers += downsampling_block(prev_in, prev_f, self.nb_l)
            prev_in = prev_f
            prev_f *= 2
        self.encoder = torch.nn.Sequential(*layers)

        #Bottleneck
        in_dim = ((self.patch//(2**self.nb_b))**2)*(prev_f//2) #last_feature_map.h * last_feature_map.w * last_feature_map.c
        self.mu = torch.nn.Linear(in_dim, self.fc)
        self.sigma = torch.nn.Linear(in_dim, self.fc)

        #Decoder
        layers = []
        for n in range(self.nb_b):
            prev_f //= 2
            next_f = prev_f // 2
            layers += upsampling_block(prev_f, next_f, self.nb_l)
        layers.append(torch.nn.Conv2d(next_f, self.in_dim, (3, 3), padding=1))
        self.decoder = torch.nn.Sequential(*layers)

    def sample_z(mu, sigma):
        eps = Variable(torch.randn(mu.size(0), self.fc))
        z = self.mu + torch.exp(self.sigma / 2) * eps

        return z

    def forward(self, x):
        x = x.view(-1, 3, self.patch, self.patch)
        x = self.encoder(x)
        x = x.view(x.size(0), -1) #Flatten x
        mu = self.mu(x)
        sigma = self.sigma(x)
        z = sample_z(mu, sigma)
        z = z.view(z.size(0), -1, self.reshape, self.reshape)
        logits = self.decoder(z)

        return logits
