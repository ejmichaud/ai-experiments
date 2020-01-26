# -*- coding: UTF-8 -*-

from math import log
from functools import reduce

import numpy as np
from sklearn.metrics import mutual_info_score
from fast_histogram import histogram2d

import torch
import torch.nn as nn


class IncompatibleArgumentError(ValueError):
    pass


def hack_range(range):
        """This version of fast_histogram handles edge cases differently
        than numpy, so we have to slightly adjust the bins."""
        d = 1e-6
        return ((range[0][0]-d, range[0][1]+d), (range[1][0]-d, range[1][1]+d))


def nats_to_bits(nats):
        r"""Convert information from nats to bits.

        Args:
            nats: float

        Returns:
            float: bits of information
        """
        return nats / log(2)


def MI(x, y, bins=32, range=((0, 1), (0, 1))):
    r"""Computes mutual information between time-series x and y.

    The mutual information between two distributions is a measure of
    correlation between them. If the distributions are independent, the
    mutual information will be 0. Mathematically, it is equivalent to the
    KL-divergence between the joint distribution and the product of the marginal
    distributions:
        
    .. math::
        I(x, y) = D_{KL}\( p(x, y) || p(x)p(y) \)

    Args:
        x (torch.tensor): a 1d tensor representing a time series of x values
        y (torch.tensor): a 1d tensor representing a time series of y values
        bins (int): the number of bins to discretize x and y values into
        range (array-like; 2x2): upper and lower values which bins can take for x and y

    Returns:
        float: the mutual information of the joint and marginal distributions
        inferred from the time series.

    TODO: implement custom version in pure pytorch without relying on sklearn
    """

    # def normalize(x):
    #     x = x / np.sum(x)
    #     x[x != x] = 0
    #     return x

    # def H(x):
    #     r = x / np.sum(x)
    #     r[r != r] = 0
    #     r = -r * np.log2(r)
    #     r[r != r] = 0
    #     return np.sum(r)
    

    assert len(x) == len(y), "time series are of unequal length"
    x = x.detach().numpy()
    y = y.detach().numpy()
    cm = histogram2d(x, y, bins=bins, range=hack_range(range))
    # return H(np.sum(cm, axis=1)) + H(np.sum(cm, axis=0)) - H(cm)
    return nats_to_bits(mutual_info_score(None, None, contingency=cm))


r"""
    The modules which are currently supported. Note that skip connections
    are currently not supported. The network structured is assumed to be
    feedforward.
"""
VALID_MODULES = {
    nn.Linear
    # nn.Conv2d: conv2d_create_matrix,
    # nn.AvgPool2d: avgpool2d_create_matrix
}

r"""
    The activations which are currently supported and their output ranges.
"""
VALID_ACTIVATIONS = {
    nn.Sigmoid: (0, 1),
    nn.Tanh: (-1, 1),
    nn.ReLU: (0, 10),
    type(None): (-10, 10)
}


def topology_of(model, input):
    r"""Get a dictionary:

    {
        `nn.Module`: {
            "input": {"activation": activation module, "shape": tuple},
            "output": {"activation": activation module, "shape": tuple}
        },
        ...
    } 
    for modules in `model` provided they are in VALID_MODULES

    Because PyTorch uses a dynamic computation graph, the number of activations
    that a given module will return is not intrinsic to the definition of the module,
    but can depend on the shape of its input. We therefore need to pass data through
    the network to determine its connectivity. 

    This function passes `input` into `model` and gets the shapes of the tensor 
    inputs and outputs of each child module in model, provided that they are
    instances of VALID_MODULES. It also finds the modules run before and after
    each child module, provided they are in VALID_ACTIVATIONS. 

    Args:
        model (nn.Module): feedforward neural network
        input (torch.tensor): a valid input to the network

    Returns:
        Dictionary of dictionaries of dictionaries (see above)
    """

    topology = {}
    hooks = []
    
    prv = None
    def register_hook(module):
        def hook(module, input, output):
            nonlocal prv
            if type(module) in VALID_MODULES:
                structure = {
                    "input": dict(),
                    "output": dict()
                }
                structure["input"]["activation"] = prv if type(prv) in VALID_ACTIVATIONS else None
                structure["input"]["shape"] = tuple(input[0].shape)
                structure["output"]["activation"] = None
                structure["output"]["shape"] = tuple(output.shape)
                topology[module] = structure
                prv = module
            if type(module) in VALID_ACTIVATIONS:
                if prv in topology:
                    topology[prv]["output"]["activation"] = module
                prv = module
        if type(module) in VALID_MODULES or type(module) in VALID_ACTIVATIONS:
            hooks.append(module.register_forward_hook(hook))

    model.apply(register_hook)
    model(input)
    for hook in hooks:
        hook.remove()
    return topology


MEMORY_LIMIT = 250000000 # 250 million floats

def _chunk_sizes(samples, num_inputs, num_outputs, limit):
    """Generator for noise tensor sizes. 

    Sometimes, the input and output matrices are too big to store
    on the GPU, so we have to divide up samples into smaller
    chunks and evaluate on them. If :
            samples * max(num_inputs, num_outputs) <= limit,
    then just yields samples. Otherwise breaks samples into
    chunks of size limit // max(num_inputs, num_outputs),
    and also yields the remainder.
    """
    width = max(num_inputs, num_outputs)
    size = limit // width
    for _ in range(size, samples+1, size):
        yield size
    if size > samples:
        yield samples
    remainder = samples % size
    if remainder and width * samples >= limit:
        yield remainder


def _indices_and_batch_sizes(samples, batch_size):
    """Generator for batch sizes and indices into noise input
    and output tensors.

    Divides `samples` into chunks of size batch_size. Yields a
    tuple of indices, and also a batch size. Includes the remainder.
    """
    if batch_size > samples:
        yield (0, samples), samples
    start, end = 0, batch_size
    for _ in range(batch_size, samples+1, batch_size):
        yield (start, end), batch_size
        start, end = end, end + batch_size
    last_batch = samples % batch_size
    if last_batch and batch_size <= samples:
        yield (samples-last_batch, samples), last_batch


def _EI_of_layer_manual_samples(layer, samples, batch_size, bins, in_shape, in_range, \
    out_shape, out_range, activation, device):
    """Helper function for EI_of_layer that computes the EI of layer `layer`
    with a set number of samples."""
    in_u, in_l = in_range
    num_inputs = reduce(lambda x, y: x * y, in_shape)
    num_outputs = reduce(lambda x, y: x * y, out_shape)

    CMs = np.zeros((num_inputs, num_outputs, bins, bins)) # histograms for each input/output pair
    if out_range == 'dynamic':
        dyn_out_ranges = np.zeros((num_outputs, 2))
        dyn_ranges_set = False

    for chunk_size in _chunk_sizes(samples, num_inputs, num_outputs, MEMORY_LIMIT):
        print(chunk_size)
        inputs = torch.zeros((chunk_size, *in_shape), device=device)
        outputs = torch.zeros((chunk_size, *out_shape), device=device)
        for (i0, i1), bsize in _indices_and_batch_sizes(chunk_size, batch_size):
            sample = (in_u - in_l) * torch.rand((bsize, *in_shape), device=device) + in_l
            inputs[i0:i1] = sample
            with torch.no_grad():
                result = activation(layer(sample))
            outputs[i0:i1] = result
        inputs = torch.flatten(inputs, start_dim=1)
        outputs = torch.flatten(outputs, start_dim=1)
        if out_range == 'dynamic' and not dyn_ranges_set:
            for B in range(num_outputs):
                out_l = torch.min(outputs[:, B]).item()
                out_u = torch.max(outputs[:, B]).item()
                dyn_out_ranges[B][0] = out_l
                dyn_out_ranges[B][1] = out_u
            dyn_ranges_set = True

        for A in range(num_inputs):
            for B in range(num_outputs):
                if out_range == 'dynamic':
                    out_r = tuple(dyn_out_ranges[B])
                else:
                    out_r = out_range
                CMs[A, B, :, :] += histogram2d(inputs[:, A].to('cpu').detach().numpy(),
                                            outputs[:, B].to('cpu').detach().numpy(),
                                            bins=bins,
                                            range=hack_range((in_range, out_r)))
    EI = 0.0
    for A in range(num_inputs):
        for B in range(num_outputs):
            EI += nats_to_bits(mutual_info_score(None, None, contingency=CMs[A, B, :, :]))
    return EI


def _EI_of_layer_auto_samples(layer, batch_size, bins, in_shape, in_range, \
    out_shape, out_range, activation, device, threshold):
    """Helper function of EI_of_layer that computes the EI of layer `layer`
    using enough samples to be within `threshold`% of the true value. 


    """
    MULTIPLIER = 2
    INTERVAL = 10000
    SAMPLES_SO_FAR = INTERVAL

    def has_converged(EIs):
        if len(EIs) < 2:
            return False
        slope = (EIs[-2] - EIs[-1]) / INTERVAL
        error = slope * SAMPLES_SO_FAR * (MULTIPLIER - 1)
        if error / EIs[-1] > threshold:
            return False
        return True
    
    in_u, in_l = in_range
    num_inputs = reduce(lambda x, y: x * y, in_shape)
    num_outputs = reduce(lambda x, y: x * y, out_shape)

    EIs = []
    CMs = np.zeros((num_inputs, num_outputs, bins, bins)) # histograms for each input/output pair
    if out_range == 'dynamic':
        dyn_out_ranges = np.zeros((num_outputs, 2))
        dyn_ranges_set = False

    while True:
        for chunk_size in _chunk_sizes(INTERVAL, num_inputs, num_outputs, MEMORY_LIMIT):
            print(chunk_size)
            inputs = torch.zeros((chunk_size, *in_shape), device=device)
            outputs = torch.zeros((chunk_size, *out_shape), device=device)
            for (i0, i1), bsize in _indices_and_batch_sizes(chunk_size, batch_size):
                sample = (in_u - in_l) * torch.rand((bsize, *in_shape), device=device) + in_l
                inputs[i0:i1] = sample
                with torch.no_grad():
                    result = activation(layer(sample))
                outputs[i0:i1] = result
            inputs = torch.flatten(inputs, start_dim=1)
            outputs = torch.flatten(outputs, start_dim=1)
            if out_range == 'dynamic' and not dyn_ranges_set:
                for B in range(num_outputs):
                    out_l = torch.min(outputs[:, B]).item()
                    out_u = torch.max(outputs[:, B]).item()
                    dyn_out_ranges[B][0] = out_l
                    dyn_out_ranges[B][1] = out_u
                dyn_ranges_set = True

            for A in range(num_inputs):
                for B in range(num_outputs):
                    if out_range == 'dynamic':
                        out_r = tuple(dyn_out_ranges[B])
                    else:
                        out_r = out_range
                    CMs[A, B, :, :] += histogram2d(inputs[:, A].to('cpu').detach().numpy(),
                                                outputs[:, B].to('cpu').detach().numpy(),
                                                bins=bins,
                                                range=hack_range((in_range, out_r)))
        EI = 0.0
        for A in range(num_inputs):
            for B in range(num_outputs):
                EI += nats_to_bits(mutual_info_score(None, None, contingency=CMs[A, B, :, :]))
        EIs.append(EI)
        print("{}: {}".format(SAMPLES_SO_FAR, EI))
        if has_converged(EIs):
            return EIs[-1]
        INTERVAL = int(SAMPLES_SO_FAR * (MULTIPLIER - 1))
        SAMPLES_SO_FAR += INTERVAL
        
        
def EI_of_layer(layer, topology, threshold=0.05, batch_size=20, bins=64, \
        samples=None, in_range=None, out_range=None, activation=None, device='cpu'):
    """Computes the effective information of neural network layer `layer`.

    Args:
        layer (nn.Module): a module in `topology`
        topology (dict): topology object (nested dictionary) returned from topology_of function
        samples (int): the number of noise samples run through `layer`
        batch_size (int): the number of samples to run `layer` on simultaneously
        bins (int): the number of bins to discretize in_range and out_range into for MI calculation
        in_range (tuple): (lower_bound, upper_bound) by default determined from `topology`
        out_range (tuple): (lower_bound, upper_bound) by default determined from `topology`
        activation (function): the output activation of `layer`, by defualt determined from `topology`
        device: 'cpu' or 'cuda' or `torch.device` instance

    Returns:
        float: an estimate of the EI of layer `layer`
    """
    
    #################################################
    #   Determine shapes, ranges, and activations   #
    #################################################
    in_shape = topology[layer]["input"]["shape"][1:]
    if in_range == 'dynamic':
        raise ValueError("Input range cannot be dynamic, only output range can be.")
    if in_range is None:
        activation_type = type(topology[layer]["input"]["activation"])
        in_range = VALID_ACTIVATIONS[activation_type]
    out_shape = topology[layer]["output"]["shape"][1:]
    if out_range is None:
        activation_type = type(topology[layer]["output"]["activation"])
        out_range = VALID_ACTIVATIONS[activation_type]

    if activation is None:
        activation = topology[layer]["output"]["activation"]
        if activation is None:
            activation = lambda x: x

    #################################################
    #             Call helper functions             #
    #################################################
    if samples is not None:
        return _EI_of_layer_manual_samples(layer=layer, 
            samples=samples, 
            batch_size=batch_size,
            bins=bins,
            in_shape=in_shape,
            in_range=in_range,
            out_shape=out_shape,
            out_range=out_range,
            activation=activation,
            device=device)
    return _EI_of_layer_auto_samples(layer=layer,
                batch_size=batch_size,
                bins=bins,
                in_shape=in_shape,
                in_range=in_range,
                out_shape=out_shape,
                out_range=out_range,
                activation=activation,
                device=device,
                threshold=threshold)


def sensitivity_of_layer(layer, topology, samples=500, batch_size=20, bins=64, \
        in_range=None, out_range=None, activation=None, device='cpu'):
    """Computes the sensitivity of neural network layer `layer`.

    Args:
        layer (nn.Module): a module in `topology`
        topology (dict): topology object (nested dictionary) returned from topology_of function
        samples (int): the number of noise samples run through `layer`
        batch_size (int): the number of samples to run `layer` on simultaneously
        bins (int): the number of bins to discretize in_range and out_range into for MI calculation
        in_range (tuple): (lower_bound, upper_bound) by default determined from `topology`
        out_range (tuple): (lower_bound, upper_bound) by default determined from `topology`
        activation (function): the output activation of `layer`, by defualt determined from `topology`
        device: 'cpu' or 'cuda' or `torch.device` instance

    Returns:
        float: an estimate of the sensitivity of layer `layer`

    TODO: add dynamic out_range support
    """
    def indices_and_batch_sizes():
        if batch_size > samples:
            yield (0, samples), samples
        start, end = 0, batch_size
        for _ in range(batch_size, samples+1, batch_size):
            yield (start, end), batch_size
            start, end = end, end + batch_size
        last_batch = samples % batch_size
        if last_batch and batch_size <= samples:
            yield (samples-last_batch, samples), last_batch
    
    #################################################
    #   Determine shapes, ranges, and activations   #
    #################################################
    in_shape = topology[layer]["input"]["shape"]
    if in_range is None:
        activation_type = type(topology[layer]["input"]["activation"])
        in_range = VALID_ACTIVATIONS[activation_type]
    out_shape = topology[layer]["output"]["shape"]
    if out_range is None:
        activation_type = type(topology[layer]["output"]["activation"])
        out_range = VALID_ACTIVATIONS[activation_type]
    in_shape, out_shape = in_shape[1:], out_shape[1:]
    in_l, in_u = in_range
    if activation is None:
        activation = topology[layer]["output"]["activation"]
        if activation is None:
            activation = lambda x: x
    
    #################################################
    #   Create buffers for layer input and output   #
    #################################################
    num_inputs = reduce(lambda x, y: x * y, in_shape)
    num_outputs = reduce(lambda x, y: x * y, out_shape)
    inputs = torch.zeros((samples, num_inputs), device=device)
    outputs = torch.zeros((samples, num_outputs), device=device)

    sensitivity = 0.0
    for A in range(num_inputs):
        for (i0, i1), size in indices_and_batch_sizes():
            sample = torch.zeros((size, num_inputs)).to(device)
            sample[:, A] = (in_u - in_l) * torch.rand((size,), device=device) + in_l
            inputs[i0:i1] = sample
            with torch.no_grad():
                result = activation(layer(sample.reshape((size, *in_shape))))
            outputs[i0:i1] = result.flatten(start_dim=1)
        for B in range(num_outputs):
            sensitivity += MI(inputs[:, A].to('cpu'), outputs[:, B].to('cpu'), bins=bins, range=(in_range, out_range))
        inputs.fill_(0)
        outputs.fill_(0)
    return sensitivity


