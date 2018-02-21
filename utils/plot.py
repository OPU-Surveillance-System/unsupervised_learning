import matplotlib.pyplot as plt
import numpy as np

def plot_auc(fpr, tpr, auc, name=None):
    """
    Plot the ROC curve]
    Args:
        fpr (numpy.array): False positive rate
        tpr (numpy.array): True positive rate
        auc (float): Area under the curve
        name (str): name to save the figure (if None: show the figure)
    """

    plt.clf()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = {:.4f})'.format(auc))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc="lower right")

    if name != None:
        plt.savefig(name, format='svg', bbox_inches='tight')
    else:
        plt.show()

def plot_reconstruction_hist(n, a, name=None):
    """
    Plot histograms of reconstruction error distributions
    Args:
        n (numpy.array): Normal distribution
        a (numpy.array): Abnormal distribution
        name (str): name to save the figure (if None: show the figure)
    """

    plt.clf()
    plt.hist(n, bins=100, alpha=0.75, label='Normal')
    plt.hist(a, bins=100, alpha=0.75, label='Abnormal')
    plt.xlabel('Reconstruction error')
    plt.ylabel('Number of frames')
    plt.legend(loc='upper left')

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
