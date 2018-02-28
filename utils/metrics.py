import torch

def per_image_error(dist, x, y):
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

def normalize_reconstruction_errors(r):
    """
    Normalize the reconstruction errors (see Hassan et al. https://arxiv.org/abs/1604.04574)
    Args:
        r (torch.tensor): Tensor containing the reconstruction errors
    """

    max_error = torch.max(r, 0)[0]
    min_error = torch.min(r, 0)[0]
    r_ = (r - min_error) / max_error

    return r_

def abnormal_score(r, d, alpha):
    """
    Compute the abnormal score by considering both the reconstruction error and the discriminator's output (see Schlegl et al. https://arxiv.org/abs/1703.05921)
    Args:
        r (torch.tensor): Tensor containing the reconstruction errors
        d (torch.tensor): Tensor containing the discriminator's ouputs
        alpha (float): Weighting constant
    """

    r_ = normalize_reconstruction_errors(r)
    score = ((1 - alpha) * r_) + (alpha * d)

    return score

def mean_image_abnormal_score(r, d, alpha, patch_size):
    """
    Compute the mean patch abnormal score per image
    Args:
        r (torch.tensor): Tensor containing the patch reconstruction errors
        d (torch.tensor): Tensor containing the patch discriminator's ouputs
        alpha (float): Weighting constant
        patch_size (int): Size of the patch
    """

    patch_abnormal_score = abnormal_score(r, d, alpha)
    reshape = score.view(-1, (256 // patch_size)**2)
    mean_image_score = torch.mean(reshape, 1)

    return mean_image_score
