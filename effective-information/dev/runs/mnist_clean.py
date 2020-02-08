# coding: utf-8

from pathlib import Path
from random import randint

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from deep_ei import topology_of, ei_of_layer, sensitivity_of_layer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
torch.set_default_dtype(dtype)

dataset_dir = Path("data")
if not dataset_dir.exists():
    dataset_dir.mkdir()

class OneHot(object):
    """Creates one-hot vector given int"""
    def __init__(self, classes, dtype):
        self.classes = classes
        self.dtype = dtype
        
    def __call__(self, label):
        label = torch.tensor(label, dtype=torch.long)
        return F.one_hot(label, self.classes).to(self.dtype)
    
class RandomOneHot(object):
    """Creates one-hot vector with random class"""
    def __init__(self, classes, dtype):
        self.classes = classes
        self.dtype = dtype
        
    def __call__(self, label):
        random_label = torch.tensor(randint(0, self.classes-1), dtype=torch.long)
        return F.one_hot(random_label, self.classes).to(self.dtype)


training_data = torchvision.datasets.MNIST(dataset_dir, 
                                train=True, 
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda img: img.reshape((784,)))
                                ]),
                                target_transform=OneHot(10, dtype)
                                )

testing_data = torchvision.datasets.MNIST(dataset_dir, 
                                train=False, 
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda img: img.reshape((784,)))
                                ]))

training_loader = torch.utils.data.DataLoader(training_data, 
                            batch_size=15,
                            shuffle=True)

testing_loader = torch.utils.data.DataLoader(testing_data, 
                            batch_size=15,
                            shuffle=False)


network = nn.Sequential(
    nn.Linear(784, 50, bias=False),
    nn.Sigmoid(),
    nn.Linear(50, 30, bias=False),
    nn.Sigmoid(),
    nn.Linear(30, 30, bias=False),
    nn.Sigmoid(),
    nn.Linear(30, 10, bias=False),
    nn.Sigmoid()
)
layer1, _, layer2, _, layer3, _, layer4, _ = network


optimizer = torch.optim.SGD(network.parameters(), lr=1e-1)
# optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss(reduction="sum")

for epoch in range(10):
    epoch_loss = 0
    for sample, target in training_loader:
        optimizer.zero_grad()
        batch_loss = loss_fn(network(sample.to(device)), target.to(device))
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    outof = 0
    accuracy = 0
    with torch.no_grad():
        for sample, labels in testing_loader:
            output = network(sample.to(device))
            _, pred = torch.max(output, 1)
            accuracy += (pred == labels.to(device)).sum().item()
            outof += len(labels)
    accuracy = accuracy / outof
    print("Epoch: {0:2d} | Loss: {1:.3f} | Accuracy: {2:.3f}".format(epoch, epoch_loss, accuracy))



