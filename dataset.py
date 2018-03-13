from torch.utils.data import Dataset, DataLoader
from scipy import misc
import os
import numpy as np

import utils.process

class VideoDataset(Dataset):
    """
    Create a dataset
    """

    def __init__(self, summary, root_dir, mode='RGB'):
        """
        VideoDataset constructor
        Args:
            summary (str): Path to a dataset summary file
            root_dir (str): Path to the dataset frames
            model (str): Image color mode (L: black and white, RGB: RGB)
        """

        self.root_dir = root_dir
        with open(summary, 'r') as f:
            content = f.read().split('\n')[:-1]
        self.frames = [os.path.join(self.root_dir, '{}'.format(c.split('\t')[0])) for c in content[0:5]]
        self.labels = [int(c.split('\t')[1]) for c in content]
        self.mode = mode

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
        if self.mode == 'L':
            img = img.reshape((256, 256, 1))
        img = utils.process.preprocess(img) #Normalize the image
        img = np.rollaxis(img, 2, 0)

        lbl = self.labels[idx]

        name = os.path.basename(self.frames[idx])

        sample = {'img': img, 'lbl': lbl, 'name': name}

        return sample
