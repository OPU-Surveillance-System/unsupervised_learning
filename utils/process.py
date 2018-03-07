def preprocess(x):
    """
    Normalize the given image between -1 and 1
    Args:
        x (numpy.array or torch.Tensor): Image of any shape (assume pixels to range in [0, 255])
    """

    x = (x - 127.5) / 127.5
    #x = x / x.max()

    return x

def deprocess(x):
    """
    Denormalize the given normalized image between 0 and 1
    Args:
        x (numpy.array or torch.Tensor): Image of any shape (assume pixels to range in [-1, 1])
    """

    x = (x + 1) * 0.5

    return x
