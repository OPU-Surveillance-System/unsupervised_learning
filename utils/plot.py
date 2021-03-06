import matplotlib.pyplot as plt
import numpy as np

def plot_mnist_likelihood_hist(n, a, name=None):
    """
    Plot histograms of reconstruction error distributions
    Args:
        n (numpy.array): Normal distribution
        a (numpy.array): Abnormal distribution
        name (str): name to save the figure (if None: show the figure)
    """

    plt.clf()
    weights = np.ones_like(n)/(len(n))
    plt.hist(n, bins=100, alpha=0.5, weights=weights, label='MNIST', color='blue')
    weights = np.ones_like(a)/(len(n))
    x2, bins2, p2 = plt.hist(a, bins=100, alpha=0.5, weights=weights, label='Alphabet', color='red')
    for item2 in p2:
        item2.set_height(item2.get_height()/sum(x2))
    plt.xlabel('Log likelihood')
    plt.ylabel('Normalized number of images')
    plt.legend(loc='upper right')

    if name != None:
        plt.savefig(name, format='svg', bbox_inches='tight')
    else:
        plt.show()

def plot_reconstruction_images(inputs, pred, name):
    """
    Plot example of reconstruction images
    Args:
        inputs (numpy.array): True images
        pred (numpy.array): Reconstructed images
        name (str): name to save the figure (if None: show the figure)
    """

    plt.clf()
    nb_plots = min(inputs.shape[0], 4)
    if inputs.shape[3] == 1:
        inputs = inputs.reshape((-1, inputs.shape[1], inputs.shape[2]))
        pred = pred.reshape((-1, inputs.shape[1], inputs.shape[2]))
    #inputs
    for i in range(nb_plots):
        ax = plt.subplot2grid((2, nb_plots), (0, i), rowspan=1, colspan=1)
        ax.imshow(inputs[i])
        ax.axis('off')
    #pred
    for i in range(nb_plots):
        ax = plt.subplot2grid((2, nb_plots), (1, i), rowspan=1, colspan=1)
        ax.imshow(np.clip(pred[i], 0.0, 1.0))
        ax.axis('off')

    if name != None:
        plt.savefig(name, format='svg', bbox_inches='tight')
    else:
        plt.show()

def plot_reconstruction_noiy_images(inputs, pred, noisy, name):
    """
    Plot example of reconstruction images
    Args:
        inputs (numpy.array): True images
        pred (numpy.array): Reconstructed images
        name (str): name to save the figure (if None: show the figure)
    """

    plt.clf()
    nb_plots = min(inputs.shape[0], 4)
    if inputs.shape[3] == 1:
        inputs = inputs.reshape((-1, 256, 256))
        pred = pred.reshape((-1, 256, 256))
        noisy = noisy.reshape((-1, 256, 256))
    #inputs
    for i in range(nb_plots):
        ax = plt.subplot2grid((3, nb_plots), (0, i), rowspan=1, colspan=1)
        ax.imshow(inputs[i])
        ax.axis('off')
    #noisy
    for i in range(nb_plots):
        ax = plt.subplot2grid((3, nb_plots), (1, i), rowspan=1, colspan=1)
        ax.imshow(noisy[i])
        ax.axis('off')
    #pred
    for i in range(nb_plots):
        ax = plt.subplot2grid((3, nb_plots), (2, i), rowspan=1, colspan=1)
        ax.imshow(pred[i])
        ax.axis('off')

    if name != None:
        plt.savefig(name, format='svg', bbox_inches='tight')
    else:
        plt.show()

def plot_generated_images(inputs, name):
    """
    Plot example of reconstruction images
    Args:

    """

    plt.clf()
    nb_plots = min(inputs.shape[0], 4)
    #inputs
    for i in range(nb_plots):
        ax = plt.subplot2grid((1, nb_plots), (0, i), rowspan=1, colspan=1)
        ax.imshow(inputs[i])
        ax.axis('off')

    if name != None:
        plt.savefig(name, format='svg', bbox_inches='tight')
    else:
        plt.show()

def plot_real_vs_fake_loss(real, fake, name):
    """
    Plot discriminator loss for real vs fake samples
    Args:
        real (list): Floats' list of length equal to the number of training epochs (real samples)
        fake (list): Floats' list of length equal to the number of training epochs (fake samples)
        name (str): name to save the figure (if None: show the figure)
    """

    plt.clf()
    x = list(range(len(real)))
    plt.plot(x, real, c='red', label='real')
    plt.plot(x, fake, c='blue', label='fake')
    plt.legend()

    if name != None:
        plt.savefig(name, format='svg', bbox_inches='tight')
    else:
        plt.show()

def plot_abnormal_score_vs_label(abnormal_score, label, name):
    """
    Plot discriminator loss for real vs fake samples
    Args:
        abnormal_score (list): Floats' list of length equal to the number of frames in the testset (abnormal score)
        fake (list): Floats' list of length equal to the number of frames in the testset (labels)
        name (str): name to save the figure (if None: show the figure)
    """

    plt.clf()
    x = list(range(len(abnormal_score)))
    plt.plot(x, abnormal_score, c='blue', label='abnormal score')
    plt.plot(x, label, c='green', label='label')
    plt.legend()

    if name != None:
        plt.savefig(name, format='svg', bbox_inches='tight')
    else:
        plt.show()

def plot_distribution(distribution, name):
    """
    """

    plt.clf()
    x = list(range(256))
    plt.plot(x, distribution)

    if name != None:
        plt.savefig(name, format='svg', bbox_inches='tight')
    else:
        plt.show()
