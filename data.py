import os
import torchvision.datasets as datasets

__DATASETS_DEFAULT_PATH = '/home/ehoffer/Datasets/'


def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True, datasets_path=__DATASETS_DEFAULT_PATH):
    train = (split == 'train')
    if datasets_path is None:
        root = os.path.join(__DATASETS_DEFAULT_PATH, name)
    else:
        root = datasets_path
    if name == 'cifar10':
        return datasets.CIFAR10(root=root,
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name == 'cifar100':
        return datasets.CIFAR100(root=root,
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'mnist':
        return datasets.MNIST(root=root,
                              train=train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'stl10':
        return datasets.STL10(root=root,
                              split=split,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'imagenet':
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        return datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=target_transform)
