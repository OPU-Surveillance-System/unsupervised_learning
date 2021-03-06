from torch.utils.data import Dataset, DataLoader
from scipy import misc
import os
import random
import numpy as np

import utils.process

class VideoDataset(Dataset):
    """
    Create a dataset
    """

    def __init__(self, summary, root_dir, mode='RGB', size='256,256', val=0):
        """
        VideoDataset constructor
        Args:
            summary (str): Path to a dataset summary file
            root_dir (str): Path to the dataset frames
            model (str): Image color mode (L: black and white, RGB: RGB)
            val (float): Ratio of elements used as validation
        """

        self.root_dir = root_dir
        with open(summary, 'r') as f:
            content = f.read().split('\n')[:-1]
        if val != 0:
            bound = int(len(content) * val)
            content = content[0:bound]
        self.frames = [os.path.join(self.root_dir, '{}'.format(c.split('\t')[0])) for c in content]
        self.labels = [int(c.split('\t')[1]) for c in content]
        self.mode = mode
        self.size = [int(s) for s in size.split(',')]

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

        img = misc.imread(self.frames[idx], mode=self.mode)
        img = misc.imresize(img, self.size)
        if self.mode == 'L':
            img = img.reshape((self.size[0], self.size[1], 1))
        img = utils.process.preprocess(img) #Normalize the image
        img = np.rollaxis(img, 2, 0)

        lbl = self.labels[idx]

        name = os.path.basename(self.frames[idx])

        sample = {'img': img, 'lbl': lbl, 'name': name, 'complete_name': self.frames[idx]}

        return sample
