"""
Torch-based K-Means
by Ali Hassani

K-Means Dataset
"""
import random
from torch.utils.data import Dataset
from .utils import squared_norm, row_norm


class KMeansDataset(Dataset):
    """
    K-Means Compatible Dataset
    """
    def __init__(self, x, metric='default', similarity_based=False):
        self.data = x
        self.data_norm = None
        if type(metric) is str and metric == 'default':
            self.data_norm = row_norm(x) if similarity_based else squared_norm(x)

    def random_sample(self, n):
        """
        Returns n random samples from the dataset.
        """
        idx = random.sample(range(self.data.size(0)), n)
        return self.__getitem__(idx)

    def __len__(self):
        return len(self.data)

    @property
    def dim(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        if self.data_norm is not None:
            return self.data[idx, :], self.data_norm[idx, :]
        return self.data[idx, :], [None for _ in range(len(idx))]
