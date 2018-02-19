import matplotlib.pyplot as plt

def plot_auc(fpr, tpr, auc, name=None):
    """
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
    inputs (numpy.array): True images (4 images)
    pred (numpy.array): Reconstructed images (4 images)
    name (str): name to save the figure (if None: show the figure)
    """

    plt.clf()
    #inputs
    ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((2, 4), (0, 1), rowspan=1, colspan=1)
    ax3 = plt.subplot2grid((2, 4), (0, 2), rowspan=1, colspan=1)
    ax4 = plt.subplot2grid((2, 4), (0, 3), rowspan=1, colspan=1)
    ax1.imshow(inputs[0])
    ax2.imshow(inputs[1])
    ax3.imshow(inputs[2])
    ax4.imshow(inputs[3])
    #pred
    ax5 = plt.subplot2grid((2, 4), (0, 0), rowspan=1, colspan=1)
    ax6 = plt.subplot2grid((2, 4), (0, 1), rowspan=1, colspan=1)
    ax7 = plt.subplot2grid((2, 4), (0, 2), rowspan=1, colspan=1)
    ax8 = plt.subplot2grid((2, 4), (0, 3), rowspan=1, colspan=1)
    ax5.imshow(pred[0])
    ax6.imshow(pred[1])
    ax7.imshow(pred[2])
    ax8.imshow(pred[3])

    if name != None:
        plt.savefig(name, format='svg', bbox_inches='tight')
    else:
        plt.show()
