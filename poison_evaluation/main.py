'''Evaluate the performance of poisons.'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
import pickle

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from PIL import Image, ImageFilter

from model import get_model
from data import get_data
from utils import get_loss_function, get_scheduler, AttackPGD
from evaluation import train, test, test_on_trainset
from options import options


args = options().parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
trainloader, testloader, clean_trainloader = get_data(args)

test_accs = []
clean_trainset_accs = []
train_accs = []
for i in range(args.runs):
    net = get_model(args, device)
    test_accs.append([])

    criterion = get_loss_function(args)

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)

    scheduler = get_scheduler(args, optimizer)

    for epoch in range(args.epochs):
        train_acc = train(args, net, trainloader, optimizer, criterion, device) 
        if epoch == (args.epochs - 1):
            train_accs.append(train_acc)

        if (epoch % 25) == 0 or epoch == (args.epochs - 1):
            test_acc, predicted = test(args, net, testloader, device)
            if epoch == args.epochs - 1:
                clean_trainset_acc, _ = test_on_trainset(
                    args, net, clean_trainloader, device) 
                clean_trainset_accs.append(clean_trainset_acc)
            test_accs[i].append(test_acc)
            print(test_accs)
        if args.dryrun:
            break
        scheduler.step()

    if args.dryrun:
        break

    print(test_accs)
    print([test_acc[-1] for test_acc in test_accs])
    final_accs = [test_acc[-1] for test_acc in test_accs]
    print(f'{args.net} Train Acc: Mean {np.mean(np.array(train_accs))}, \
        Std_error: {np.std(np.array(train_accs))/np.sqrt(args.runs)}')
    print(f'{args.net} Clean Trainset: \
        Mean {np.mean(np.array(clean_trainset_accs))},\
        Std_error: {np.std(np.array(clean_trainset_accs))/np.sqrt(args.runs)}')
    print(f'{args.net} Mean {np.mean(np.array(final_accs))}, \
        Std_error: {np.std(np.array(final_accs))/np.sqrt(args.runs)}')
