from ei import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dtype = torch.cuda.float if torch.cuda.is_available() else torch.float
dtype = torch.float32
torch.set_default_dtype(dtype)
print(f"Using device: {device}")


layer = nn.Linear(784, 10, bias=False).to(device)
top = topology_of(layer, torch.zeros(1, 784))
for s in [10, 100, 1000, 5000, 10000, 30000]:
    print("samples: {}".format(s))
    print(EI_of_layer(layer, top, samples=s, batch_size=40, bins=16, device=device))

