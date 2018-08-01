from torch.utils.data import Dataset, DataLoader
from scipy import misc
import os
import random
import numpy as np

import utils.process

class UnsupervisedDataset(Dataset):
    """
    Create a dataset
    """

    def __init__(self, n_summary, a_summary, root_dir, ratio):
        """
        VideoDataset constructor
        Args:
            n_summary (str): Path to a (normal) dataset summary file
            a_summary (str): Path to a (abnormal) dataset summary file
            root_dir (str): Path to the dataset frames
            ratio (float): Ratio of abnormal patterns
        """

        with open(n_summary, 'r') as f:
            n_content = f.read().split('\n')[:-1]
        with open(a_summary, 'r') as f:
            a_content = f.read().split('\n')[:-1]

        random.shuffle(n_content)
        random.shuffle(a_content)

        dataset_size = len(n_content)
        to_remove = int((ratio * dataset_size) / 100)

        n_set = [n_content[random.randint(0, len(n_content) - 1)] for i in range(dataset_size - to_remove)]
        a_set = [a_content[random.randint(0, len(a_content) - 1)] for i in range(to_remove)]
        dataset = n_set + a_set

        self.root_dir = root_dir
        self.frames = [os.path.join(self.root_dir, '{}'.format(c.split('\t')[0])) for c in dataset]
        self.labels = [int(c.split('\t')[1]) for c in dataset]

    def __len__(self):
        """
        Return the dataset length
        """

        return len(self.frames)

    def __getitem__(self, idx):
        """
        Return the designated item
        Args:
            idx (int): item index
        """

        img = misc.imread(self.frames[idx], mode='L')
        img = misc.imresize(img, [64, 64])
        img = img.reshape((64, 64, 1))
        img = utils.process.preprocess(img) #Normalize the image
        img = np.rollaxis(img, 2, 0)

        lbl = self.labels[idx]

        name = os.path.basename(self.frames[idx])

        sample = {'img': img, 'lbl': lbl, 'name': name}

        return sample

# ratios = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 40, 50]
# for r in ratios:
#     ds = UnsupervisedDataset('data/summaries/umn_normal_trainset', 'data/summaries/umn_abnormal_trainset', '/home', r)
