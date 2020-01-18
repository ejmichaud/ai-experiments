# -*- coding: UTF-8 -*-

from math import log

import numpy as np
from sklearn.metrics import mutual_info_score

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    assert len(x) == len(y), "time series are of unequal length"
    x = x.detach().numpy()
    y = y.detach().numpy()
    contingency_matrix, _, _ = np.histogram2d(x, y, bins=bins, range=range)
    return nats_to_bits(mutual_info_score(None, None, contingency=contingency_matrix))


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


def topology_of(model, input):
    r"""Get a dictionary {module: (in_shape, out_shape), ...} for modules in `model`.

    Because PyTorch uses a dynamic computation graph, the number of activations
    that a given module will return is not intrinsic to the definition of the module,
    but can depend on the shape of its input. We therefore need to pass data through
    the network to determine its connectivity. 

    This function passes `input` into `model` and gets the shapes of the tensor 
    inputs and outputs of each child module in model, provided that they are
    instances of VALID_MODULES.

    Args:
        model (nn.Module): feedforward neural network
        input (torch.tensor): a valid input to the network

    Returns:
        Dictionary {`nn.Module`: tuple(in_shape, out_shape)}
    """

    shapes = {}
    hooks = []
    
    def register_hook(module):
        def hook(module, input, output):
            shapes[module] = (tuple(input[0].shape), tuple(output.shape))
        if type(module) in VALID_MODULES:
            hooks.append(module.register_forward_hook(hook))

    model.apply(register_hook)
    model(input)
    for hook in hooks:
        hook.remove()
    return shapes


def EI_of_layer(layer, topology, samples=30000, batch_size=20, bins=32, device='cpu', activation=torch.sigmoid):
    """This should allow for the easy calculation of each layer's EI."""
    
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
    
    in_shape, out_shape = topology[layer]
    in_shape, out_shape = in_shape[1:], out_shape[1:]
    inputs = torch.zeros((samples, *in_shape), device=device)
    outputs = torch.zeros((samples, *out_shape), device=device)

    for (i0, i1), size in indices_and_batch_sizes():
        sample = torch.rand((size, *in_shape), device=device)
        inputs[i0:i1] = sample
        result = activation(layer(sample))
        outputs[i0:i1] = result

    inputs = torch.flatten(inputs, start_dim=1)
    outputs = torch.flatten(outputs, start_dim=1)
    num_inputs, num_outputs = inputs.shape[1], outputs.shape[1]
    EI = 0.0
    for A in range(num_inputs):
        for B in range(num_outputs):
            EI += MI(inputs[:, A].to('cpu'), outputs[:, B].to('cpu'), bins=bins)
    return EI




