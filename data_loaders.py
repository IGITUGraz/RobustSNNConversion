from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


def cifar10(args):
    norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    if args.cutout:
        transform_train.transforms.append(Cutout(n_holes=1, length=8, norm_mean=norm[0]))
    transform_test = transforms.Compose([transforms.ToTensor()])
    train_dataset = CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=True, download=True, transform=transform_train)
    val_dataset = CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=False, download=True, transform=transform_test)
    return train_dataset, val_dataset, norm, 10


def cifar100(args):
    norm = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    if args.cutout:
        transform_train.transforms.append(Cutout(n_holes=1, length=8, norm_mean=norm[0]))
    transform_test = transforms.Compose([transforms.ToTensor()])
    train_dataset = CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), train=True, download=True, transform=transform_train)
    val_dataset = CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), train=False, download=True, transform=transform_test)
    return train_dataset, val_dataset, norm, 100


def svhn(args):
    norm = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    if args.cutout:
        transform_train.transforms.append(Cutout(n_holes=1, length=8, norm_mean=norm[0]))
    transform_test = transforms.Compose([transforms.ToTensor()])
    train_dataset = SVHN(root=os.path.join(args.data_dir, 'SVHN'), split='train', transform=transform_train)
    val_dataset = SVHN(root=os.path.join(args.data_dir, 'SVHN'), split='test', transform=transform_test)
    return train_dataset, val_dataset, norm, 10


def tinyimagenet(args):
    norm = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform_train = transforms.Compose([transforms.RandomResizedCrop(64),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    if args.cutout:
        transform_train.transforms.append(Cutout(n_holes=1, length=16, norm_mean=norm[0]))
    transform_test = transforms.Compose([transforms.ToTensor()])
    train_dataset = ImageFolder(root=os.path.join(args.data_dir, 'tinyimagenet', 'train'), transform=transform_train)
    val_dataset = ImageFolder(root=os.path.join(args.data_dir, 'tinyimagenet', 'val'), transform=transform_test)
    return train_dataset, val_dataset, norm, 200


class Cutout(object):
    def __init__(self, n_holes, length, norm_mean):
        self.n_holes = n_holes
        self.length = length
        self.mean = norm_mean

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        img[0] = img[0] + ((1 - mask[0]) * self.mean[0])
        img[1] = img[1] + ((1 - mask[1]) * self.mean[1])
        img[2] = img[2] + ((1 - mask[2]) * self.mean[2])

        return img
