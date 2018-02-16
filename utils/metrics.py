def get_per_image_error(dist, x, y):
    """
    Return the l2 norm for each image in a batch
    Args:
        dist (torch.nn.Module): Distance function
        x (torch.Tensor): prediction
        y (torch.Tensor): target
    """

    x_ = x.view(x.size(0), -1)
    y_ = y.view(y.size(0), -1)
    e = dist(x_, y_)

    return e
