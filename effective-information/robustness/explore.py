#!/usr/bin/env python
# coding: utf-8

# ///////////////////////////////////////////
#                   IMPORTS
# ///////////////////////////////////////////

import os
from pathlib import Path
from itertools import islice
import gzip
import pickle
from hashlib import sha1

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

from foresight.ei import ei


# ///////////////////////////////////////////
#                 LOAD DATASET
# ///////////////////////////////////////////

dir_path = Path().absolute()
dataset_path = dir_path.parent.parent / "data/mnist.pkl.gz"
if not dataset_path.exists():
    print('Downloading dataset with curl ...')
    if not dataset_path.parent.exists():
        os.mkdir(dataset_path.parent)
    url = 'http://ericjmichaud.com/downloads/mnist.pkl.gz'
    os.system('curl -L {} -o {}'.format(url, dataset_path))
print('Download failed') if not dataset_path.exists() else print('Dataset acquired')
f = gzip.open(dataset_path, 'rb')
mnist = pickle.load(f)
f.close()
print('Loaded data to variable `mnist`')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dtype = torch.cuda.float if torch.cuda.is_available() else torch.float
dtype = torch.float32
torch.set_default_dtype(dtype)

print(f"Using device: {device}")



# ///////////////////////////////////////////
#               DEFINE `Dataset`
# ///////////////////////////////////////////
class MNISTDataset(Dataset):
    """MNIST Digits Dataset."""
    def __init__(self, data, transform=None):
        """We save the dataset images as torch.tensor since saving 
        the dataset in memory inside a `Dataset` object as a 
        python list or a numpy array causes a multiprocessiing-related 
        memory leak."""
        self.images, self.labels = zip(*data)
        self.images = torch.from_numpy(np.array(self.images)).to(dtype)
        self.labels = torch.tensor(np.argmax(self.labels, axis=1)).to(torch.long)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        if self.transform:
            image, label = self.transform((image, label))
        return image, label


# ///////////////////////////////////////////
#               DEFINE MODEL
# ///////////////////////////////////////////
class FullyConnected(nn.Module):
    """Single-hidden-layer dense neural network."""
    def __init__(self, activation, use_bias, layers):
        super(FullyConnected, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1], bias=use_bias))

    def forward(self, x):
        if len(layers) == 1:
            return F.log_softmax(self.layers[0](x), dim=1)
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        return F.log_softmax(self.layers[i+1](x), dim=1)


# ///////////////////////////////////////////
#            INITIALIZE DATA LOADERS
# ///////////////////////////////////////////
train_data = MNISTDataset(mnist[:60000])
test_data = MNISTDataset(mnist[60000:])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=True)


# ///////////////////////////////////////////
#               DEFINE PARAMETERS
# ///////////////////////////////////////////
initializers = {
    'kaiming': None, # (default)
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
}

biases = {'True': True, 'False': False}

activations = {'tanh': torch.tanh, 'relu': torch.relu}

topologies = {
    '784 -> 10': [28*28, 10],
    '784 -> 50 -> 10': [28*28, 50, 10],
    '784 -> 100 -> 10': [28*28, 100, 10],
    '784 -> 250 -> 10': [28*28, 250, 10],
    '784 -> 100 -> 100 -> 10': [28*28, 100, 100, 10],
    '784 -> 300 -> 100 -> 10': [28*28, 300, 100, 10]
}

TOTAL_CONFIGS = len(initializers) * len(biases) * len(activations) * len(topologies)

def weight_initializer(name):
    def init_weights(m):
        if name == 'kaiming':
            return
        if isinstance(m, nn.Linear):
            initializers[name](m.weight)
    return init_weights


def generate_data(topology, use_bias, activation, initializer):
    """Takes in a set of parameters and executes a training run.

    Args:
        topology: (list) of ints specifying size of each layer
        use_bias: (bool)
        activation: (torch.*) activation function to use
        initializer: (str): name of initializer to call weight_initializer on

    Returns:
        None

    Prints:
        Various messages

    Creates:
        Folder and saves .pkl file into it
    """
    hasher = sha1()
    hasher.update(pickle.dumps([topology, bias, activation, initializer]))
    foldername = str(hasher.hexdigest())
    parent = Path().absolute()
    if not (parent/'runs').exists():
        (parent/"runs").mkdir()
    runs = parent/"runs"
    if (runs/foldername).exists():
        print(f"""The folder for the following parameters ALREADY EXISTS:
    topology: f{topology}
    bias: f{use_bias}
    activation: f{activation}
    initializer: f{initializer}""")
        return
    (runs/foldername).mkdir()
    folder = runs/foldername

    model = FullyConnected(
            activations[activation],
            biases[bias],
            topologies[topology]
        ).to(device)
    model.apply(weight_initializer(initializer))
    
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # --- DATA TO COLLECT FOR EACH REALIZATION ---
    num_batches_data = []
    eis_data = []
    losses_data = [] #[(train, test), (train, test), ...]
    accuracies_data = []

    num_batches = 0
    for epoch in range(80):
        for sample, target in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(sample.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
            num_batches += 1
            
            if num_batches % 50 == 0:
                with torch.no_grad():
                    input = next(iter(test_loader))[0].to(device)
                    EI = ei(model, input=input, device=device)

                outof = 0
                loss = 0
                with torch.no_grad():
                    for x, labels in islice(test_loader, 0, 500): # 500 batches of 20 samples
                        output = model(x.to(device))
                        loss += loss_fn(output, labels.to(device)).item()
                        _, pred = torch.max(output, 1)
                        outof += len(labels)
                test_loss = loss / outof

                outof = 0
                loss = 0
                with torch.no_grad():
                    for x, labels in islice(train_loader, 0, 500): # 500 batches of 20 samples
                        output = model(x.to(device))
                        loss += loss_fn(output, labels.to(device)).item()
                        _, pred = torch.max(output, 1)
                        outof += len(labels)
                training_loss = loss / outof

                outof = 0
                accuracy = 0
                with torch.no_grad():
                    for x, labels in test_loader:
                        output = model(x.to(device))
                        _, pred = torch.max(output, 1)
                        accuracy += (pred == labels.to(device)).sum().item()
                        outof += len(labels)
                accuracy = accuracy / outof

                num_batches_data.append(num_batches)
                eis_data.append(EI)
                losses_data.append((training_loss, test_loss))
                accuracies_data.append(accuracy)
        print("EPOCH {:3d}: ACC {:0.3f} | EI {:2.3f}".format(epoch, accuracies_data[-1], eis_data[-1]))
    
    # done training
    params = (topology, bias, activation, initializer)
    data = (params, (num_batches_data, eis_data, losses_data, accuracies_data))
    with open(folder/"data.pkl", "wb") as f:
        pickle.dump(data, f)


count = 0
for topology in topologies:
    for bias in biases:
        for activation in activations:
            for initializer in initializers:
                count += 1
                print(f"""
>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<
STARTING RUN {count}/{TOTAL_CONFIGS} WITH PARAMETERS:
topology: f{topology}
bias: f{bias}
activation: f{activation}
initializer: f{initializer}
>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<""")
                generate_data(topology, bias, activation, initializer)








