import torch
import torch.nn as nn

from ei import EI_of_layer, topology_of

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dtype = torch.cuda.float if torch.cuda.is_available() else torch.float
dtype = torch.float32
torch.set_default_dtype(dtype)
print(f"Using device: {device}")

WIDTH = 70
layer = nn.Linear(WIDTH, WIDTH, bias=False).to(device)
top = topology_of(layer, torch.zeros(1, WIDTH).to(device))

for s in [100, 1000, 10000, 30000, 60000, 100000, 200000, 500000, 1000000]:
    print("samples: {}".format(s))
    print(EI_of_layer(layer, top, samples=s, batch_size=100, bins=16, device=device))

