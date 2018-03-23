import torch

class GatedPixelCnn(torch.nn.Module):
    def __init__(self, in_dim, nb_f, nb_l, nb_b, fc=None):
