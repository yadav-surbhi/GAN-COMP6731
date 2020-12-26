import numpy as np
import torch
from torchvision import datasets, transforms
import os
import errno
import matplotlib.pyplot as plt
from torch.autograd.variable import Variable

DATABASE = os.getcwd() + "/Database"
MODELS_PATH = os.getcwd() + "/Models/"
IMAGES_FOLDER = os.getcwd() + "/Images"


def load_MNIST(batch_size, d=-1):

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    trainset = datasets.MNIST(DATABASE, download=True, train=True, transform=transform)

    if d != -1:
        idx = trainset.targets == d
        trainset.targets = trainset.targets[idx]
        trainset.data = trainset.data[idx]

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    return train_loader


def load_CIFAR10(batch_size):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = datasets.CIFAR10(DATABASE, download=True, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    return train_loader


def visualize_data(loader, number=10):
    iterator = iter(loader)
    images, labels = iterator.next()
    figure = plt.figure()
    for index in range(1, number + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
        # plt.title("{}".format(labels[index]))
    plt.show()


def display_image(images, epoch, n_batch, d=-1):
    figure = plt.figure()
    for index in range(1, images.size(0) + 1):
        plt.subplot(int(images.size(0) / 10), 10, index)
        plt.axis('off')
        plt.imshow(images[index - 1].numpy().squeeze(), cmap='gray_r')

    plt.show()

    make_dir(IMAGES_FOLDER + "/MNIST")
    image_file_name = '{}/epoch_{}_batch_{}.png'.format(IMAGES_FOLDER + "/MNIST", epoch, n_batch)
    if epoch == 0 and n_batch == 0:
        image_file_name = IMAGES_FOLDER + '/MNIST_pic.png'
    if d != -1:
        make_dir(IMAGES_FOLDER + "/MNIST" + str(d))
        image_file_name = '{}/epoch_{}_batch_{}.png'.format(IMAGES_FOLDER + "/" + str(d), epoch, n_batch)
    figure.savefig(image_file_name)


def images_to_vectors(images):
    return images.view(images.size(0), 784)


def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


def noise(size):
    """
    Generates a 1-d vector of gaussian sampled random values
    """
    n = Variable(torch.randn(size, 100))

    return n


def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
