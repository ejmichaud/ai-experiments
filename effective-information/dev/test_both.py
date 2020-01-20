import torch
import torch.nn as nn

from ei import EI_of_layer, topology_of
from TimeSeriesEI import CalculateNetworkEI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dtype = torch.cuda.float if torch.cuda.is_available() else torch.float
dtype = torch.float32
torch.set_default_dtype(dtype)
print(f"Using device: {device}")


network = nn.Sequential(nn.Linear(784, 10, bias=False), nn.Sigmoid()).to(device)
layer, _ = network
top = topology_of(layer, torch.zeros(1, 784).to(device))

for s in [100, 1000, 10000, 30000, 60000, 100000]:
    print("samples: {}".format(s))
    print("MY WAY:")
    print(EI_of_layer(layer, top, samples=s, batch_size=100, bins=16, device=device))
    print("THEIR WAY:")
    input = torch.zeros((s, 784)).to(device)
    print(CalculateNetworkEI(network, input, size=s, bins=16, device=device))

