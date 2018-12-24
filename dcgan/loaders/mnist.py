import gzip
import logging
import os

import torch
import numpy as np
import requests

from torch.utils.data import Dataset

from defs import APP_DIR

logging.basicConfig()
logger = logging.getLogger('mnist')
logger.setLevel(logging.INFO)


class DatasetNotFoundException(Exception):
    pass

class DownloadFailedException(Exception):
    pass


class MNISTDataset(Dataset):
    """
    Dataset for the popular MNIST dataset
    """
    def __init__(self, root_dir, train=False, download=False, transform=None):
        assert isinstance(root_dir, str)

        self.root_dir = os.path.join(APP_DIR, root_dir)
        self.train = train
        self.download = download
        self.transform = transform

        # Check for the existence of the dataset in the root_dir
        if not self._is_exists():
            if not self.download:
                raise DatasetNotFoundException('The dataset directory could not be found. \
                                                Did you specify the right path in root_dir parameter ?. \
                                                Optionally set the `download` option to True to download the dataset')
        
            # Download the dataset otherwise
            self.download_mnist(root_dir)
        
        # Load the dataset here
        logger.info('Parsing binary files and creating dataset')
        self.training_images, self.training_labels = (
            self._parse_image_file(filename=os.path.join(self.root_dir, 'train-images-idx3-ubyte')),
            self._parse_label_file(filename=os.path.join(self.root_dir, 'train-labels-idx1-ubyte'))
        )
        self.test_images, self.test_labels = (
            self._parse_image_file(filename=os.path.join(self.root_dir,'t10k-images-idx3-ubyte')),
            self._parse_label_file(filename=os.path.join(self.root_dir,'t10k-labels-idx1-ubyte'))
        )

    def __len__(self):
        if self.train:
            return self.training_images.shape[0]
        return self.test_images.shape[0]

    def __getitem__(self, index):
        sample = {}
        if self.train:
            sample = (
                self.training_images[index, :],
                self.training_labels[index]
            )
        else:
            sample = (
                self.test_images[index, :],
                self.test_labels[index]
            )
        if not self.transform:
            return sample
        
        return (self.transform(sample[0]), sample[1])

    @property
    def _download_urls(self):
        return {
            'train-images-idx3-ubyte':'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte':'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte':'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte':'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        }

    def _is_exists(self):
        """
        Checks if the MNIST dataset is already downloaded in the specified location
        """
        for file_name, _ in self._download_urls.items():
            if not os.path.isfile(file_name):
                return False

    def download_mnist(self, loc):
        """
        Downloads the MNIST dataset
        """
        # Check if the download location exists
        if not os.path.isdir(loc):
            if os.path.isfile(loc):
                raise OSError('The download location cannot be a file')

            logger.info('The specified location does not exist. Creating..')
            os.makedirs(loc, exist_ok=True)
        
        for file_name, url in self._download_urls.items():
            zipped_download_loc = os.path.join(loc, file_name + '.gz')
            unzipped_download_loc = os.path.join(loc, file_name)

            if os.path.isfile(unzipped_download_loc):
                continue
            
            if not os.path.isfile(zipped_download_loc):
                # Download the file
                logger.info('Downloading %s', zipped_download_loc)
                response = requests.get(url)
                if not (response.status_code == 200):
                    raise DownloadFailedException(f'The file {file_name} could not be downloaded')
            
                # Write the response data to the specified file
                with open(zipped_download_loc, 'wb') as writer:
                    writer.write(response.content)
            
            logger.debug('Unzipping %s', zipped_download_loc)
            with gzip.open(zipped_download_loc, 'rb') as data:
                content = data.read()
                with open(unzipped_download_loc, 'wb') as writer:
                    writer.write(content)
        
        logger.info('Download and Extraction completed.')

    def _parse_image_file(self, filename):
        """
        Parses the MNIST image binary file and returns a numpy array
        """
        data = []
        with open(filename, 'rb') as reader:
            data = reader.read()

        # For the file format refer to `http://yann.lecun.com/exdb/mnist/`
        magic_number = int.from_bytes(data[:4], byteorder='big')
        assert magic_number == 2051
        num_images = int.from_bytes(data[4:8], byteorder='big')
        num_rows = int.from_bytes(data[8:12], byteorder='big')
        num_cols = int.from_bytes(data[12:16], byteorder='big')
        
        dataset = np.frombuffer(data, dtype=np.uint8, offset=16)
        dataset = np.reshape(dataset, newshape=(num_images, 1, num_rows, num_cols))

        return torch.from_numpy(dataset).float()

    def _parse_label_file(self, filename):
        """
        Parses the MNIST label binary file and returns a numpy array
        """
        data = []
        with open(filename, 'rb') as reader:
            data = reader.read()
        
        magic_number = int.from_bytes(data[:4], byteorder='big')
        assert magic_number == 2049
        labels = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(labels).float()
