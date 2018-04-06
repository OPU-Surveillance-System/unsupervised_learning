import torch

class MaskedConvolution(torch.nn.Conv2d):
    def __init__(self, in_dim, out_dim, kernel_size, mask_type, padding):
        super(MaskedConvolution, self).__init__(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.mask_type = mask_type

        self.register_buffer('mask', self.weight.data.clone())
        _, _, height, width = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, height // 2, width // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, height // 2 + 1:] = 0

        #Weights initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight)

    def forward(self, x):
        self.weight.data *= self.mask
        x = super(MaskedConvolution, self).forward(x)

        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, h):
        super(ResidualBlock, self).__init__()

        self.h = h

        layers = []
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv2d(2 * self.h, self.h, (1, 1)))
        layers.append(torch.nn.BatchNorm2d(self.h))
        layers.append(torch.nn.ReLU())
        layers.append(MaskedConvolution(self.h, self.h, (3, 3), 'B', 1))
        layers.append(torch.nn.BatchNorm2d(self.h))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv2d(self.h, 2 * self.h, (1, 1)))
        layers.append(torch.nn.BatchNorm2d(2 * self.h))
        self.layers = torch.nn.Sequential(*layers)

        #Weights initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight)

    def forward(self, x):
        y = self.layers(x)
        x = x + y

        return x

class PixelCNN(torch.nn.Module):
    def __init__(self, h, n, d):
        super(PixelCNN, self).__init__()

        self.h = h
        self.n = n
        self.d = d

        self.first_layer = MaskedConvolution(1, 2 * self.h, (7, 7), 'A', 3)
        self.bn = torch.nn.BatchNorm2d(2 * self.h)
        self.residual_blocks = torch.nn.Sequential(*[ResidualBlock(self.h) for n in range(self.n + 1)])
        self.top_layer = torch.nn.Sequential(*[torch.nn.ReLU(), torch.nn.Conv2d(2 * self.h, self.d, (1, 1)), torch.nn.BatchNorm2d(self.d), torch.nn.ReLU()])
        #self.top_layer = torch.nn.Sequential(*[torch.nn.ReLU(), torch.nn.Conv2d(2 * self.h, self.d, (1, 1)), torch.nn.ReLU()])
        self.evidence = torch.nn.Conv2d(self.d, 256, (1, 1))

        #Weights initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.bn(x)
        x = self.residual_blocks(x)
        x = self.top_layer(x)
        logits = self.evidence(x)
        output = torch.nn.functional.softmax(logits, dim=1)

        return logits, output
