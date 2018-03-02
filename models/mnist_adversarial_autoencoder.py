import torch
import math

def downsampling_block(in_dim, nb_f, nb_l):
    layers = []
    for n in range(nb_l):
        layers.append(torch.nn.Conv2d(in_dim, nb_f, (3, 3), padding=1))
        layers.append(torch.nn.SELU())
        in_dim = nb_f
    layers.append(torch.nn.MaxPool2d((2, 2), (2, 2)))

    return layers

def upsampling_block(in_dim, nb_f, nb_l):
    layers = [torch.nn.ConvTranspose2d(in_dim, nb_f, (2, 2), (2, 2))]
    layers.append(torch.nn.SELU())
    for n in range(nb_l):
        layers.append(torch.nn.Conv2d(nb_f, nb_f, (3, 3), padding=1))
        layers.append(torch.nn.SELU())

    return layers

class Encoder(torch.nn.Module):
    def __init__(self, nb_f, nb_l, nb_b, fc, input_size=256):
        super(Encoder, self).__init__()
        self.in_dim = 1
        self.nb_f = nb_f
        self.nb_l = nb_l
        self.nb_b = nb_b
        self.fc = fc
        self.ips = input_size

        #Conv part
        layers = []
        in_dim = self.in_dim
        nb_f = self.nb_f
        for b in range(self.nb_b):
            layers += downsampling_block(in_dim, nb_f, self.nb_l)
            in_dim = nb_f
            nb_f *= 2
        self.conv = torch.nn.Sequential(*layers)

        #Latent part
        self.flatten = ((self.ips//(2**self.nb_b))**2)*in_dim #last_feature_map.h * last_feature_map.w * last_feature_map.c
        self.out_dim = self.flatten
        self.latent = torch.nn.Linear(self.flatten, self.fc)

        #Weights initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.)*math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.in_features))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = x.view((-1, self.flatten))
        x = self.latent(x)

        return x

class Decoder(torch.nn.Module):
    def __init__(self, encoder_flatten, nb_l, nb_b, fc, img_size=256):
        super(Decoder, self).__init__()
        self.nb_l = nb_l
        self.nb_b = nb_b
        self.fc = fc
        self.encoder_flatten = encoder_flatten
        self.img_size = img_size

        #Latent part
        layers = []
        layers.append(torch.nn.Linear(self.fc, self.encoder_flatten))
        layers.append(torch.nn.SELU())
        self.latent = torch.nn.Sequential(*layers)

        #Conv part
        layers = []
        self.h = self.img_size // (2**self.nb_b)
        self.in_dim = self.encoder_flatten // (self.h**2)
        in_dim = self.in_dim
        for b in range(self.nb_b):
            nb_f = in_dim // 2
            layers += upsampling_block(in_dim, nb_f, self.nb_l)
            in_dim = nb_f
        layers.append(torch.nn.Conv2d(in_dim, 1, (3, 3), (1, 1), padding=1))
        self.conv = torch.nn.Sequential(*layers)
        self.pred = torch.nn.Tanh()

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
        x = self.latent(x)
        x = x.view((-1, self.in_dim, self.h, self.h))
        logits = self.conv(x)
        pred = self.pred(logits)

        return logits, pred

class Discriminator(torch.nn.Module):
    def __init__(self, latent_size, layers):
        super(Discriminator, self).__init__()
        self.layers = [int(l) for l in layers.split(',')]
        self.latent_size = latent_size

        layers = []
        s = self.latent_size
        for l in self.layers:
            layers.append(torch.nn.Linear(s, l))
            layers.append(torch.nn.SELU())
            s = l
        layers.append(torch.nn.Linear(s, 1))
        self.fc = torch.nn.Sequential(*layers)
        self.pred = torch.nn.Tanh()

        #Weights initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.in_features))
                m.bias.data.zero_()

    def forward(self, x):
        logits = self.fc(x)
        pred = self.pred(logits)

        return logits, pred
