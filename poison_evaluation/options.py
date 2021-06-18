"""Implement an ArgParser for main.py (poison evaluation) ."""

import argparse

def options():
    parser = argparse.ArgumentParser(description='Argparser for sanity check')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--runs', default=5, type=int, help='num runs')
    parser.add_argument('--epochs', default=100, type=int, help='num epochs')
    parser.add_argument('--net', default='ResNet', type=str)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--scheduler', type=str, default='linear')
    parser.add_argument('--criterion', type=str, default='')
    parser.add_argument('--dataset_type', default='load', type=str, choices=['load', 'subset', 'clean'])
    parser.add_argument('--baseset', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR100',
                            'SVHN'])
    parser.add_argument('--dryrun', action='store_true')
    return parser
