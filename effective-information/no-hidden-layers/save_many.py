#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pathlib import Path
from itertools import islice
import gzip
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

from foresight.ei import ei


# In[ ]:


dir_path = Path().absolute()
dataset_path = dir_path.parent / "data/mnist.pkl.gz"
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


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dtype = torch.cuda.float if torch.cuda.is_available() else torch.float
dtype = torch.float32
torch.set_default_dtype(dtype)


# In[ ]:


device


# In[ ]:


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


# In[ ]:


class SoftmaxRegression(nn.Module):
    """Single-layer softmax network."""
    def __init__(self, n_in, n_out):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(n_in, n_out, bias=False)
    
    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=1)


# ## No Hidden Layer Softmax Network

# In[ ]:


train_data = MNISTDataset(mnist[:60000])
test_data = MNISTDataset(mnist[60000:])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=True)


# In[ ]:


NUM_REALIZATIONS = 10

# --- ALL DATA ---
graph_data = []


for realization in range(NUM_REALIZATIONS):
    print("Starting realization {} of {}".format(realization, NUM_REALIZATIONS))
    
    model = SoftmaxRegression(28*28, 10).to(device)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # --- DATA TO COLLECT FOR EACH REALIZATION ---
    num_batches_data = []
    eis_data = []
    losses_data = [] #[(train, test), (train, test), ...]
    accuracies_data = []

    def update_metrics():
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

        if num_batches % 1500 == 0:
            print("Epoch: {:3d} | EI: {:2.3f} | Accuracy: {:0.3f}".format(epoch, EI, accuracy))

    # --- TRAIN ---

    num_batches = 0
    for epoch in range(80):
        for sample, target in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(sample.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
            num_batches += 1
            if num_batches % 100 == 0:
                update_metrics()
    print("Appending data for realization {}".format(realization))
    graph_data.append((num_batches_data, eis_data, losses_data, accuracies_data))
    
import pickle
with open("plots/graph_data_many.pkl", "wb") as f:
    pickle.dump(graph_data, f)

