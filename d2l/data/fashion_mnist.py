import os
import sys
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt


def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
        '~', '.pytorch', 'datasets', 'fashion-mnist')):
    """Download the Fashion-MNIST dataset and then load into memory."""

    train_data_path = '/Users/yingshuaihao/PycharmProjects/Dataset/FashionMNIST/processed/training.pt'
    test_data_path = '/Users/yingshuaihao/PycharmProjects/Dataset/FashionMNIST/processed/test.pt'

    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        X, y = torch.load(train_data_path)
        X = X.float()
        mnist_train = TensorDataset(X, y)

        X, y = torch.load(test_data_path)
        X = X.float()
        mnist_test = TensorDataset(X, y) 
    else:
        root = os.path.expanduser(root)
        transformer = []
        if resize:
            transformer += [transforms.Resize(resize)]
        transformer += [transforms.ToTensor()]
        transformer = transforms.Compose(transformer)

        mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, transform=transformer, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, transform=transformer, download=True)

    num_workers = 0 if sys.platform.startswith('win32') else 4

    train_iter = DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


def get_fashion_mnist_labels(labels):
    """Get text labels for Fashion-MNIST."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    """Plot Fashion-MNIST images with labels."""
    from IPython import display
    display.set_matplotlib_formats('svg')

    # use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
