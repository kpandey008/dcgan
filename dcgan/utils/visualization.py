'''
Useful script for data and result visualization purposes
'''
import math

import numpy as np
import torch

from matplotlib import pyplot as plt

def plot_training_images(dataset, num_images, img_shape, shuffle=False):
    """
    Plots some images from the dataset
    """
    assert isinstance(img_shape, tuple)

    indices = range(num_images)
    if shuffle:
        indices = np.random.randint(0, len(dataset), size=(num_images,))

    # determine the number of rows and columns required for the grid
    num_rows = num_cols = math.ceil(math.sqrt(num_images))

    # create a new figure
    fig = plt.figure(figsize=(num_rows, num_cols))
    plt.axis('off')
    fig.suptitle('Training images')
    
    img_index = 1
    for index in indices:
        sample = dataset[index]

        plt.subplot(num_rows, num_cols, img_index)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.imshow(np.reshape(sample['image'].numpy(), newshape=img_shape), cmap="gray")

        img_index += 1

    plt.show()
