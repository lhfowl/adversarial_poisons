'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from consts import *

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def get_loss_function(args):
    if args.criterion == '':
        criterion = nn.CrossEntropyLoss()
    elif 'kl' in args.criterion:
        def kl_loss(outputs, targets):
            return torch.nn.functional.kl_div(F.log_softmax(outputs, dim=1),F.softmax(targets, dim=1))
        criterion = kl_loss

    return criterion


def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                      milestones=[args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142], gamma=0.1)
    return scheduler



class AttackPGD(nn.Module):
    def __init__(self, basic_net, args):
        super(AttackPGD, self).__init__()
        dm, ds = get_stats(args)
        dm, ds = dm.to('cuda'), ds.to('cuda')
        config = {
            'epsilon': 8.0/255.0,
            'num_steps': 10,
            'step_size': 2.0/255.0,
            'dm': dm,
            'ds': ds
        }
        self.config = config
        self.basic_net = basic_net
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']

    def forward(self, inputs, targets):
        if self.epsilon == 0:
            return self.basic_net(inputs)
        else:
            x = inputs.detach()
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon) / self.config['ds']
            for i in range(self.num_steps):
                x.requires_grad_()
                with torch.enable_grad():
                    loss = F.cross_entropy(self.basic_net(x), targets, reduction='sum')
                grad = torch.autograd.grad(loss, [x])[0]
                x = x.detach() + self.step_size / self.config['ds'] * torch.sign(grad.detach())
                x = torch.min(torch.max(x, inputs - self.epsilon / self.config['ds']), inputs + self.epsilon / self.config['ds'])
                x = torch.max(torch.min(x, (1 - self.config['dm']) / self.config['ds']), -self.config['dm'] / self.config['ds'])
            return self.basic_net(x)

def get_stats(args):
    if args.baseset == 'CIFAR10':
        dm = torch.tensor(cifar10_mean)[None, :, None, None]
        ds = torch.tensor(cifar10_std)[None, :, None, None]
        return dm, ds
    else:
        raise NotImplementedError
