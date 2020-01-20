import torch
import torch.nn
import numpy
import sklearn
import sklearn.metrics
import scipy
import scipy.stats

import collections

# Initialization Types
Uniform = "uniform"
Standard = "standard"
CompressedRandom = "c_rand"
UniformRandom = "u_rand"
ExtendedRandom = "e_rand"
DualGaussian = "dual_gaussian"
MaximumEI = "max"

# Noise Types
WhiteNoise = "white_noise"
Brownian = "brownian"

# Creates Brownian Motion (Weiner Process) centered on zero with given statistics
# x0: initial condition (starting point, always zero)
# dt: the time interval over which variance is measured (basically, spikiness)
# dL: the variance over interval dt (basically, amplitude)
# n: the number of points
def BrownianMotion(dt, dL, n):
    samples = scipy.stats.norm.rvs(size = (n,), scale = dL*numpy.sqrt(dt))
    motion = numpy.cumsum(samples)
    return motion

# Cylinderize a time series by gluing the top and bottom bounds together.
def Cylinderize(motion,center,lower,upper):
    span = upper - lower
    return (((motion * span) + (center - lower)) % (span)) + lower

# Calculate the binned mutual information between two time series.
# This is a discrete binning, for now.
def MI(x, y, bins, range = [[0.0,1.0],[0.0,1.0]]):
    binned = numpy.histogram2d(x, y, bins, range = range)[0]
    return numpy.log2(numpy.e) * sklearn.metrics.mutual_info_score(None,None,contingency=binned)
    #return sklearn.metrics.mutual_info_score(None,None,contingency=binned) / numpy.log(2.0)

# Create a tensor of the given shape composed of timeseries of brownian motion
# TODO: make alpha and sigma? values passable parameters
def CreateBrownianTensor(shape, size, dtype = torch.float, device = torch.device("cpu")):
    newshape = (size,) + shape
    data = numpy.zeros(newshape)
    prod = numpy.prod(numpy.array([x for x in shape]))
    flat = data.reshape((prod,size))
    
    for place in range(prod):
            center = numpy.random.random()
            dat = Cylinderize(BrownianMotion(0.01,0.2,size), center, 0.0, 1.0)
            flat[place,:] = dat
    
    back = flat.reshape(newshape)
    newTensor = torch.tensor(back, device = device, dtype = dtype)
    return newTensor

# Create a tensor of the given shape composed of timeseries of white noise
def CreateWhiteNoiseTensor(shape, size, dtype = torch.float, device = torch.device("cpu")):
    newshape = (size,) + shape
    newTensor = torch.rand(newshape, dtype = dtype, device = device)
    return newTensor

# Extract the layers from the module by counting which modules have weights.
# Store the layer start, end, and size. These will be used to calculate the EI.
def ExtractLayers(network):
    place = 0
    store = None
    layers = []
    for module in network.modules():
        # Only look at modules that have a dedicated linear weight matrix. Which is only the torch.nn.modules.Linear()
        for name, param in module.named_parameters():
            if name in ["weight"]:
                if store != None:
                    start, startSize, endSize = store
                    layers.append((start, startSize, place - 1, endSize))
                store = place, param.shape[1], param.shape[0]
                break
        place += 1

    # The output layer?
    # TODO: maybe don't measure this one...
    if store != None:
        start, startSize, endSize = store
        layers.append((start, startSize, place - 1, endSize))

    return layers

# Pre and post-processing hooks for the noise injection.
# This will inject noise into the network as is, perserving generality beyond that of a simple feed-forward.
def InjectionPreHook(module, readTensor, inputTensor, size, dtype, device, *, motion = WhiteNoise):
    inputTensor = inputTensor[0]

    # Create an injection tensor time-series to continue
    injectionSize = inputTensor.shape[1]

    noiseTensor = None
    if motion == WhiteNoise:
        noiseTensor = CreateWhiteNoiseTensor((injectionSize,), size, dtype = dtype, device = device)
    elif motion == Brownian:
        noiseTensor = CreateBrownianTensor((injectionSize,), size, dtype = dtype, device = device)

    copyTensor = noiseTensor.clone().detach()
    injection = copyTensor.view((size, injectionSize))

    # Copy white noise to the read buffer
    readTensor[:] = injection.clone()
    
    # TODO: use the documented return measures of overriding input instead of doing so directly!
    inputTensor[:] = injection.clone()
    return None
    # BUG! TORCH DOCUMENTATION LIES! It claims that it is possible to change the input by returning from the hook,
    # but that turns out to be impossible in practice. Perhaps the lifetime of the injection tensor is being handled
    # incorrectly, but the most likely explanation is that the documentation claims a capability that is not present
    # on our current PyTorch implementation. As such, the above code is needed instead of this commented out code.
    #return (injection.clone(),)

def OutputPostHook(module, readTensor, inputTensor, outputTensor):
    inputTensor = inputTensor[0]
    outputTensor = outputTensor

    # Copy output into the read buffer
    readTensor[:] = outputTensor.clone()

    # No change to the output tensor
    return None

def CreateTimeSeriesBuffers(layers, size, dtype, device):
    # Create buffers to store time series tensors
    timeSeriesBuffers = []
    for layer in layers:
        start, startSize, end, endSize = layer
        injection = torch.zeros((size, startSize), dtype = dtype, device = device).clone().detach()
        output = torch.zeros((size, endSize), dtype = dtype, device = device).clone().detach()
        timeSeriesBuffers.append((injection,output))
    return timeSeriesBuffers

def AttachNetworkHooks(network, layers, buffers, size, dtype, device, *, motion = WhiteNoise):
    # Return handles that can be used to remove the hooks
    handles = []

    # Create an iterator for the network
    layerIter = iter(layers)

    # Attach hooks to the network
    try:
        place = 0
        layerPlace = 0
        start, startSize, end, endSize = next(layerIter)
        for module in network.modules():
            if place == start:
                # We register a pre-hook to inject the noise.
                # we need a nested lambda to dynamically register the hook to the appropriate buffer
                handle = module.register_forward_pre_hook(
                    (lambda i:
                        (lambda x, y: InjectionPreHook(x, buffers[i][0], y, size, dtype, device, motion = motion))
                    )(layerPlace)
                )
                handles.append(handle)

            if place == end:
                # We register a post-hook to read the output tensor
                # we need a nested lambda to dynamically register the hook to the appropriate buffer
                handle = module.register_forward_hook(
                    (lambda i: lambda x, y, z: OutputPostHook(x, buffers[i][1], y, z))(layerPlace)
                )
                handles.append(handle)

                # Extract the next layer after this one is consumed
                start, startSize, end, endSize = next(layerIter)
                layerPlace += 1

            place += 1
    except StopIteration as e:
        pass


    return handles

def CalculateNetworkEI(N, I, size = 1000, bins = 32,
                       *, dtype = torch.float, device = torch.device("cpu"), motion = WhiteNoise):
    network = N
    input = I

    # Extract layers of weights from the network
    layers = ExtractLayers(network)
    # Create the neccesary buffers to hold time series information
    buffers = CreateTimeSeriesBuffers(layers, size, dtype, device)
    # Attach the neccesary hooks to read data from a forward pass into the TimeSeriesBuffers
    handles = AttachNetworkHooks(N, layers, buffers, size, dtype, device, motion = motion)

    # Run the network, with the injection and read hooks. This will fill the TimeSeriesBuffers!
    output = network(input)

    # Remove the hooks so the network remains uncorrupted. Gradiant data will be clobbered, howerver :/
    for handle in handles:
        handle.remove()

    # Calculate per-layer EI using the data in each buffer.
    totalEI = 0.0
    layerEIs = []
    for buff in buffers:
        injection, output = buff
        layerEI = CalculateLayerEI(injection, output, bins)
        totalEI += layerEI
        layerEIs.append(layerEI)

    return totalEI, layerEIs


# TODO: this is a terrible hack!
def CreateFeedForwardNetwork(layers):
    topology = collections.OrderedDict()
    for layer in layers:
        inputIndex, inputSize, outputIndex, outputSize = layer
        topology[str(inputIndex)] = torch.nn.modules.Linear(inputSize, outputSize, bias = False)
        topology[str(outputIndex)] = torch.nn.modules.Sigmoid()

    return torch.nn.modules.Sequential(topology)
    
# TODO: This is a terrible hack as well!
def InitMaxEIModule(module, dtype, device):
    for param, item in module.named_parameters():
        if param in ["weight"]:
            size = item.shape[0]
            if item.shape[0] == item.shape[1]:
                newParams = torch.tensor(numpy.diag(numpy.ones(size) * numpy.e), dtype = dtype, device = device)
                module.weight.data[:] = newParams

# TODO: All of these initializations will not work with different network parameters
# Initiates a network with weights that are clustered randomly around positive and negative e.
# This initialization saturates the EI of individual weights in the network, preventing information gain from
# just strengthening average connectiveness..
# TODO: This may be more general than I give it credit for...
def InitRandomSaturatedModule(module, dtype, device):
    for param, item in module.named_parameters():
        if param in ["weight"]:
            shape = item.shape
            # Create a new weight tensor with saturated average weights
            # randn is a normal around zero with SD = 1.
            newWeights = torch.randn(shape, dtype = dtype, device = device)
            # Offset the normals to either e or -e
            offsets = torch.tensor(numpy.random.choice([-numpy.e, numpy.e], size = shape),
                dtype = dtype,
                device = device)
            newWeights += offsets
            # Update the weights
            module.weight.data[:] = newWeights

def InitRandomUniformModule(module, dtype, device, scale = 1.0):
    for param, item in module.named_parameters():
        if param in ["weight"]:
            shape = item.shape
            # Create a new weight tensor with uniform weights
            # Between 0 and 1
            weights = numpy.random.random(size=shape)
            weights *= 2
            weights -= 1
            weights *= scale
            newWeights = torch.tensor(weights, dtype = dtype, device = device)
            # Update the weights
            module.weight.data[:] = newWeights

def InitRandomExtendedModule(module, dtype, device):
    InitRandomUniformModule(module, dtype, device, scale = 2 * numpy.e)

def InitRandomCompressedModule(module, dtype, device):
    InitRandomUniformModule(module, dtype, device, scale = 0.01)
    
def InitStandardModule(module, dtype, device):
    name = module.__class__.__name__
    if name.find('Linear') != -1:
        # get the number of the inputs
        n = module.in_features
        scale = 1.0 / numpy.sqrt(n)
        InitRandomUniformModule(module, dtype, device, scale = scale)

def InitUniformModule(module, dtype, device):
    for param, item in module.named_parameters():
        if param in ["weight"]:
            shape = item.shape
            # Create a new weight tensor filled with weights of .5
            newWeights = torch.tensor(numpy.full(shape, 0.5), dtype = dtype, device = device)
            # Update the weights
            module.weight.data[:] = newWeights


def ScaleModuleWeights(module, dtype, device, scale):
    name = module.__class__.__name__
    if scale == Standard:
        if name.find('Linear') != -1:
            # get the number of the inputs
            n = module.in_features
            scale = 1.0 / numpy.sqrt(n)
    else: pass
    for param, item in module.named_parameters():
        if param in ["weight"]:
            shape = item.shape
            # Create a new weight tensor with uniform weights
            # Between 0 and 1
            weights = module.weight.data[:].clone().detach()
            weights *= scale
            # Update the weights
            module.weight.data[:] = weights

def ScaleNetworkWeights(network, dtype = torch.float, device = torch.device("cpu"), scale = 1.0):  
    network.apply(lambda x: ScaleModuleWeights(x, dtype, device, scale))


# TODO: again, this is a patchwork solution for now
def InitNetwork(network, initalg, dtype = torch.float, device = torch.device("cpu")):
    alg = None
    if initalg == "standard":
        alg = InitStandardModule
    elif initalg == "uniform":
        alg = InitUniformModule
    elif initalg == "c_rand":
        alg = InitRandomCompressedModule
    elif initalg == "u_rand":
        alg = InitRandomUniformModule
    elif initalg == "e_rand":
        alg = InitRandomExtendedModule
    elif initalg == "dual_gaussian":
        alg = InitRandomSaturatedModule
    elif initalg == "max":
        alg = InitMaxEIModule
    else:
        alg = lambda x,y,z: InitRandomUniformModule(x, y, z, scale = initalg)
    '''
    else:
        raise ValueError("Unknown initialization type " + initalg)
    '''
    network.apply(lambda x: alg(x, dtype, device))

# Manually calculate the maximum EI of a network by constructing a linear pipeline
def CalculateNetworkMaxEIManual(N, I, size = 1000, bins = 32, dtype = torch.float, device = torch.device("cpu")):
    network = N
    input = I
    
    # Extract the layers of weights
    layers = ExtractLayers(network)
    # Create a new network of the appropriate shape
    # TODO: this is very fragile, and probably a bad idea...
    cloned = CreateFeedForwardNetwork(layers)
    InitNetwork(cloned,"max")

    return CalculateNetworkEI(cloned, I, size, bins, dtype, device)

# For this, just calculate the number of neurons and multiply by ln(bins)
# This is just a theoretical calculation
# TODO: this is specific to sigmoid mechanisms!
def CalculateNetworkMaxEI(network, bins = 32):
    # Need the total number of output neurons in the network
    neurons = 0
    # To do so, search through each layer
    layers = ExtractLayers(network)
    # The number of output neurons is the number that can be given a non-degenerate connection!
    for layer in layers:
        # Layer has the form (inputIndex, inputSize, outputIndex, outputSize)
        neurons += layer[3]
    # Return the completed calculation, the max for each neuron is the natural log of the bin size!
    print(neurons, numpy.log(bins))
    maxEI = neurons * numpy.log(bins)
    return maxEI

# Calculate the EffectiveInformation of a feedforward neural network using pytorch nn modules
def CalculateLayerEI(injection, output, bins):
    totalEI = 0.0
    injSize = injection.shape[1]
    outSize = output.shape[1]

    totalEI = 0.0
    for i in range(injSize):
        for o in range(outSize):
           pmi = MI(injection[:,i].to('cpu').detach().numpy(), output[:,o].to('cpu').detach().numpy(), bins)
           totalEI += pmi

    return totalEI

if __name__ == "__main__":
    width = 30
    network = torch.nn.Sequential(
        torch.nn.modules.Linear(width, width, bias = False),
        torch.nn.modules.Sigmoid(),
        torch.nn.modules.Linear(width, width, bias = False),
        torch.nn.modules.Sigmoid(),
        torch.nn.modules.Linear(width, width, bias = False),
        torch.nn.modules.Sigmoid(),
        torch.nn.modules.Linear(width, width, bias = False),
        torch.nn.modules.Sigmoid()
    )

    # TODO: copy the input over the time series steps so that we don't have to do the manual matching of sizes!
    size = 10000
    bins = 64
    input = torch.zeros((size, width))

    layers = ExtractLayers(network)
    copy = CreateFeedForwardNetwork(layers)

    print(network)
    print(layers)
    print(copy)

    print(CalculateNetworkEI(network, input, size = size, bins = bins))
    InitNetwork(network, "e_rand", dtype = torch.float, device = torch.device("cpu"))
    #print(CalculateNetworkEI(copy, input, size = size, bins = bins))
    print(CalculateNetworkEI(network, input, size = size, bins = bins))
    InitNetwork(network, "u_rand", dtype = torch.float, device = torch.device("cpu"))
    print(CalculateNetworkEI(network, input, size = size, bins = bins))
    InitNetwork(network, "dual_gaussian", dtype = torch.float, device = torch.device("cpu"))
    print(CalculateNetworkEI(network, input, size = size, bins = bins))

    #print(CalculateNetworkMaxEI(network))
    #print(CalculateNetworkMaxEIManual(network, input, size = size, bins = bins))

    

