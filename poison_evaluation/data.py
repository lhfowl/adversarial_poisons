import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import random
import os
from PIL import Image
import pickle

class _CIFAR_load(torch.utils.data.Dataset):
    def __init__(self, root, baseset, dummy_root='~/data', split='train', download=False, **kwargs):

        self.baseset = baseset
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010))])
        self.transform = transform_train
        self.samples = os.listdir(os.path.join(root, 'data'))
        self.root = root

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        true_index = int(self.samples[idx].split('.')[0])
        true_img, label = self.baseset[true_index]
        return self.transform(Image.open(os.path.join(self.root, 'data',
                                            self.samples[idx]))), label

def _dataset_picker(args, clean_trainset):
    if args.dataset_type == 'load':
        trainset = _CIFAR_load(f'{args.load_path}', clean_trainset)

    elif args.dataset_type == 'clean':
        trainset = clean_trainset

    else:
        raise NotImplementedError

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    return trainset, trainloader

def _baseset_picker(args):
    if args.baseset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        clean_trainset = torchvision.datasets.CIFAR10(
            root='~/data', train=True, download=True, transform=transform_train)
        clean_trainloader = torch.utils.data.DataLoader(
            clean_trainset, batch_size=128, shuffle=False, num_workers=2)

    else:
        raise NotImplementedError

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='~/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return clean_trainset, clean_trainloader, testset, testloader

def get_data(args):
    print('==> Preparing data..')
    clean_trainset, clean_trainloader, testset, testloader = _baseset_picker(args)

    trainset, trainloader = _dataset_picker(args, clean_trainset)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, clean_trainloader
