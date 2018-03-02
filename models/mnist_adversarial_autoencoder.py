import torch
import math

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
                layers.append(torch.nn.SELU())
                in_dim = nb_f
            layers.append(torch.nn.MaxPool2d((2, 2), (2, 2)))

            return layers

        #Conv part
        layers = []
        prev_in = 1
        prev_f = self.nb_f
        for n in range(self.nb_b):
            layers += downsampling_block(prev_in, prev_f, self.nb_l)
            prev_in = prev_f
            prev_f *= 2
        self.conv = torch.nn.Sequential(*layers)
        self.last_map_dim = ','.join([str(prev_f//2), str(self.patch//(2**self.nb_b)), str(self.patch//(2**self.nb_b))])

        #Bottleneck
        flatten = ((self.patch//(2**self.nb_b))**2)*(prev_f//2) #last_feature_map.h * last_feature_map.w * last_feature_map.c
        self.bottleneck = torch.nn.Linear(flatten, self.latent_size)

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
        x = x.view(-1, 1, self.patch, self.patch)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)

        return x

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
                layers.append(torch.nn.SELU())
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
        layers.append(torch.nn.Conv2d(prev_f, 1, (3, 3), padding=1))
        self.conv = torch.nn.Sequential(*layers)

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
        x = self.bottleneck(x)
        x = x.view(-1, self.encoder_dim[0], self.encoder_dim[1], self.encoder_dim[2])
        x = self.conv(x)

        return x

################################################################################

class Discriminator(torch.nn.Module):
    def __init__(self, latent_size, layers):
        super(Discriminator, self).__init__()
        self.latent_size = latent_size
        self.layers = [int(l) for l in layers.split(',')]

        layers = []
        in_dim = self.latent_size
        for l in range(len(self.layers)):
            layers.append(torch.nn.Linear(in_dim, self.layers[l]))
            layers.append(torch.nn.SELU())
            in_dim = self.layers[l]
        layers.append(torch.nn.Linear(in_dim, 1))
        self.net = torch.nn.Sequential(*layers)

        self.activation = torch.nn.Sigmoid()

        #Weights initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.in_features))
                m.bias.data.zero_()

    def forward(self, x):
        logits = self.net(x)
        pred = self.activation(logits)

        return logits, pred
