import os

from torch.utils.data import Dataset

class DatasetNotFoundException(Exception):
    pass

class MnistLoader(Dataset):
    """
    DataLoader for the popular MNIST dataset
    """
    def __init__(self, root_dir, download=False, transform=None):
        assert isinstance(root_dir, str)
        
        self.download_urls = [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        ]
        self.root_dir = root_dir
        self.transform = transform
        self.download = download

        # Check for the existence of the dataset in the root_dir
        if not self._is_exists():
            if not self.download:
                raise DatasetNotFoundException('The dataset directory could not be found. \
                                                Did you specify the right path in root_dir parameter ?')
        
            # Download the dataset otherwise
            self.download_mnist()
        
        # Load the dataset here

    def _is_exists(self):
        pass

    def download_mnist(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass
