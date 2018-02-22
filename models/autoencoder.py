import torch
import math

class Autoencoder(torch.nn.Module):
    def __init__(self, nb_f, nb_l, nb_b, fc=None, input_size=256, rate=0.5):
        super(Autoencoder, self).__init__()
        self.in_dim = 3
        self.nb_f = nb_f
        self.nb_l = nb_l
        self.nb_b = nb_b
        self.fc = fc
        self.ips = input_size
        self.rate

        def downsampling_block(in_dim, nb_f, nb_l, rate):
            layers = []
            for n in range(nb_l):
                layers.append(torch.nn.Conv2d(in_dim, nb_f, (3, 3), padding=1))
                layers.append(torch.nn.Dropout2d(p=rate))
                layers.append(torch.nn.SELU())
                in_dim = nb_f
            layers.append(torch.nn.MaxPool2d((2, 2), (2, 2)))

            return layers

        def upsampling_block(in_dim, nb_f, nb_l, rate):
            layers = [torch.nn.ConvTranspose2d(in_dim, nb_f, (2, 2), (2, 2))]
            layers.append(torch.nn.Dropout2d(p=rate))
            layers.append(torch.nn.SELU())
            for n in range(nb_l):
                layers.append(torch.nn.Conv2d(nb_f, nb_f, (3, 3), padding=1))
                layers.append(torch.nn.Dropout2d(p=rate))
                layers.append(torch.nn.SELU())

            return layers

        def build_bottleneck(in_dim, h_dim, rate):
            layers = [torch.nn.Linear(in_dim, h_dim)]
            layers.append(torch.nn.Dropout2d(p=rate))
            layers.append(torch.nn.Linear(h_dim, in_dim))
            layers.append(torch.nn.Dropout2d(p=rate))
            layers.append(torch.nn.SELU())

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
            in_dim = ((self.ips//(2**self.nb_b))**2)*(prev_f//2) #last_feature_map.h * last_feature_map.w * last_feature_map.c
            layers = build_bottleneck(in_dim, self.fc, self.rate)
            self.bottleneck = torch.nn.Sequential(*layers)

        #Decoder
        layers = []
        for n in range(self.nb_b):
            prev_f //= 2
            next_f = prev_f // 2
            layers += upsampling_block(prev_f, next_f, self.nb_l, self.rate)
        layers.append(torch.nn.Conv2d(next_f, self.in_dim, (3, 3), padding=1))
        layers.append(torch.nn.Dropout2d(p=rate))
        self.decoder = torch.nn.Sequential(*layers)

        self.activation = torch.nn.Tanh()

        #Weights initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.)*math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.in_features))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        if self.fc:
            x = x.view(x.size(0), -1)
            x = self.bottleneck(x)
            reshape = self.ips//(2**self.nb_b)
            x = x.view(x.size(0), -1, reshape, reshape)
        logits = self.decoder(x)
        activation = self.activation(logits)

        return logits, activation
