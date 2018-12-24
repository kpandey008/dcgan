import os

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from defs import APP_DIR
from utils.config import ConfigLoader

from .mnist import MNISTDataset

config_file_path = os.path.join(APP_DIR, 'config.conf')
config = ConfigLoader(config_file_path)

# create the mnist DataLoader
root_dir = config.get_param_value('data', 'dataroot')
batch_size = int(config.get_param_value('train', 'batch_size'))
num_workers = int(config.get_param_value('data', 'workers'))

mnist_dataset = MNISTDataset(root_dir, train=True, download=True, transform=transforms.Compose([
    transforms.Normalize((0.5,), (0.5,)),
]))
mnist_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
