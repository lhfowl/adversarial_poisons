import torch
import numpy as np

# Training
def train(args, net, trainloader, optimizer, criterion, device, attack=False):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if attack:
            outputs = net(inputs, targets)
        else:
            outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        if 'kl' in args.criterion:
            _, targets = targets.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if args.dryrun:
            break
    return 100.*correct/total


def test(args, net, testloader, device, proportion=False, attack=False):
    net.eval()
    correct = 0
    total = 0
    predicted_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if attack:
                outputs = net(inputs, targets)
            else:
                outputs = net(inputs)
            _, predicted = outputs.max(1)
            predicted_labels.append(predicted)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if args.dryrun:
                break
    return 100.*correct/total, predicted_labels

def test_on_trainset(args, net, clean_trainloader, device, attack=False):
    net.eval()
    correct = 0
    total = 0
    predicted_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(clean_trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if attack:
                outputs = net(inputs, targets)
            else:
                outputs = net(inputs)
            _, predicted = outputs.max(1)
            predicted_labels.append(predicted)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if args.dryrun:
                break
    return 100.*correct/total, predicted_labels
