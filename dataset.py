from torch.utils.data import Dataset, DataLoader
from scipy import misc
import os

import utils.process

class VideoDataset(Dataset):
    """
    Create a dataset
    """

    def __init__(self, summary, root_dir):
        """
        VideoDataset constructor
        Args:
            summary (str): Path to a dataset summary file
            root_dir (str): Path to the dataset frames
        """

        self.root_dir = root_dir
        with open(summary, 'r') as f:
            content = f.read().split('\n')[:-1]
        self.frames = [os.path.join(self.root_dir, '{}.png'.format(c.split('\t')[0])) for c in content]
        self.labels = [int(c.split('\t')[1]) for c in content]

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

        img = misc.imread(self.frames[idx])
        img = utils.process.preprocess(img) #Normalize the image

        lbl = self.labels[idx]

        name = os.path.basename(self.frames[idx])

        sample = {'img': img, 'lbl': lbl, 'name': name}

        return sample

ds = VideoDataset('/home/scom/Documents/opu_surveillance_system/anomaly_detection/data/umn_trainset_labels', '/home/scom/Documents/opu_surveillance_system/anomaly_detection/data/')
