import torch
import math

class Autoencoder(torch.nn.Module):
    def __init__(self, in_dim, nb_f, nb_l, nb_b, fc=None, rate=0.5):
        super(Autoencoder, self).__init__()
        self.in_dim = in_dim
        self.nb_f = nb_f
        self.nb_l = nb_l
        self.nb_b = nb_b
        self.fc = fc
        self.rate = rate

        def downsampling_block(in_dim, nb_f, nb_l, rate):
            layers = []
            for n in range(nb_l):
                layers.append(torch.nn.Conv2d(in_dim, nb_f, (3, 3), padding=1))
                layers.append(torch.nn.Dropout2d(p=rate))
                layers.append(torch.nn.ReLU())
                #layers.append(torch.nn.SELU())
                in_dim = nb_f
            layers.append(torch.nn.MaxPool2d((2, 2), (2, 2)))

            return layers

        def upsampling_block(in_dim, nb_f, nb_l, rate):
            #layers = [torch.nn.ConvTranspose2d(in_dim, nb_f, (2, 2), (2, 2))]
            layers = [torch.nn.Upsample(scale_factor=2, mode='bilinear')]
            # layers.append(torch.nn.ReLU())
            for n in range(nb_l):
                layers.append(torch.nn.Conv2d(in_dim, nb_f, (3, 3), padding=1))
                layers.append(torch.nn.Dropout2d(p=rate))
                layers.append(torch.nn.ReLU())
                #layers.append(torch.nn.SELU())
                in_dim = nb_f

            return layers

        def build_bottleneck(in_dim, h_dim):
            layers = [torch.nn.Linear(in_dim, h_dim)]
            layers.append(torch.nn.Linear(h_dim, in_dim))

            return layers


        #Encoder
        layers = []
        prev_in = self.in_dim
        prev_f = self.nb_f
        for n in range(self.nb_b):
            layers += downsampling_block(prev_in, prev_f, self.nb_l, self.rate)
            prev_in = prev_f
            prev_f *= 2
        self.encoder = torch.nn.Sequential(*layers)

        #Bottleneck
        if self.fc:
            in_dim = ((256//(2**self.nb_b))**2)*(prev_f//2) #last_feature_map.h * last_feature_map.w * last_feature_map.c
            layers = build_bottleneck(in_dim, self.fc)
            self.bottleneck = torch.nn.Sequential(*layers)

        #Decoder

        layers = []
        for n in range(self.nb_b):
            prev_f //= 2
            next_f = prev_f // 2
            layers += upsampling_block(prev_f, next_f, self.nb_l, self.rate)
        layers.append(torch.nn.Conv2d(next_f, self.in_dim, (3, 3), padding=1))
        self.decoder = torch.nn.Sequential(*layers)

        #Weights initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                #torch.nn.init.kaiming_normal(m.weight)
                torch.nn.init.xavier_uniform(m.weight, math.sqrt(2))

    def forward(self, x):
        x = self.encoder(x)
        if self.fc:
            x = x.view(x.size(0), -1)
            x = self.bottleneck(x)
            reshape = 256//(2**self.nb_b)
            x = x.view(x.size(0), -1, reshape, reshape)
        logits = self.decoder(x)

        return logits
