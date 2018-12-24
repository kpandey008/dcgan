'''
Useful script for data and result visualization purposes
'''
import math

import numpy as np
import torch

from matplotlib import pyplot as plt

def plot_training_images(dataset, num_images, img_shape, shuffle=False):
    """
    Plots some sample training images from the dataset
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

def plot_images(images):
    """
    Plots some sample images in a grid.
    images [Pytorch tensor] : Image batch in the format : B x C x H x W
    """
    num_images = images.size(0)
    shape = (images.size(3), images.size(2))
    # determine the number of rows and columns required for the grid
    num_rows = num_cols = math.ceil(math.sqrt(num_images))

    # create a new figure
    fig = plt.figure(figsize=(num_rows, num_cols))
    plt.axis('off')
    fig.suptitle('Training images')

    for image_idx in range(num_images):
        plt.subplot(num_rows, num_cols, image_idx + 1)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.imshow(np.reshape(images[image_idx, :].numpy(), newshape=shape), cmap="gray")

    plt.show()
