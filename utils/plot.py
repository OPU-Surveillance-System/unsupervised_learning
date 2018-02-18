import matplotlib.pyplot as plt

def plot_auc(fpr, tpr, auc, name=None):
    """
    fpr: False positive rate
    tpr: True positive rate
    auc (float): Area under the curve
    name (str): name to save the figure (if None: show the figure)
    """

    plt.clf()
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = {:.4f})'.format(auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
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
    plt.hist(n, bins=100, alpha=0.75)
    plt.hist(a, bins=100, alpha=0.75)
    if name != None:
        plt.savefig(name, format='svg', bbox_inches='tight')
    else:
        plt.show()
