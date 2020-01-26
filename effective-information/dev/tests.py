import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from ei import MI, _sample_sizes, _indices_and_batch_sizes


#######################################
#               MI Tests              #
#######################################
def test_MI_0():
    x = torch.tensor([0.3, 0.2, 0.4, 0.7])
    y = torch.tensor([0.6, 0.7, 0.2, 0.3])
    correct_MI = 0.31127812445913294
    measured_MI = MI(x, y, bins=2)
    error = 1e-6
    assert correct_MI - error <= measured_MI <= correct_MI + error

def test_MI_1():
    x = torch.tensor([0.0, 0.111, 0.45, 0.9])
    y = torch.tensor([0.6, 1.0, 0.2, 0.3])
    correct_MI = 0.31127812445913294
    measured_MI = MI(x, y, bins=2)
    error = 1e-6
    assert correct_MI - error <= measured_MI <= correct_MI + error

def test_MI_2():
    x = torch.tensor([0.0, 0.0, 1.0, 1.0])
    y = torch.tensor([1.0, 1.0, 0.0, 0.0])
    correct_MI = 1.0
    measured_MI = MI(x, y, bins=2)
    error = 1e-6
    assert correct_MI - error <= measured_MI <= correct_MI + error

def test_MI_3():
    x = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    y = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    correct_MI = 1.0
    measured_MI = MI(x, y, bins=2)
    error = 1e-6
    assert correct_MI - error <= measured_MI <= correct_MI + error

def test_MI_4():
    x = torch.tensor([0.0, 0.3, 0.6, 1.0])
    y = torch.tensor([0.3, 0.6, 1.0, 0.0])
    correct_MI = 2.0
    measured_MI = MI(x, y, bins=4)
    error = 1e-6
    assert correct_MI - error <= measured_MI <= correct_MI + error


#######################################
#        _sample_sizes tests          #
#######################################
def test_sample_sizes_0():
    samples = 20
    num_inputs = 4
    limit = 10
    correct_sequence = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    generated_sequence = list(_sample_sizes(samples, num_inputs, limit))
    assert len(correct_sequence) == len(generated_sequence)
    for i in range(len(correct_sequence)):
        assert correct_sequence[i] == generated_sequence[i]

def test_sample_sizes_1():
    samples = 10
    num_inputs = 3
    limit = 10
    correct_sequence = [3, 3, 3, 1]
    generated_sequence = list(_sample_sizes(samples, num_inputs, limit))
    assert len(correct_sequence) == len(generated_sequence)
    for i in range(len(correct_sequence)):
        assert correct_sequence[i] == generated_sequence[i]

def test_sample_sizes_2():
    samples = 11
    num_inputs = 5
    limit = 14
    correct_sequence = [2, 2, 2, 2, 2, 1]
    generated_sequence = list(_sample_sizes(samples, num_inputs, limit))
    assert len(correct_sequence) == len(generated_sequence)
    for i in range(len(correct_sequence)):
        assert correct_sequence[i] == generated_sequence[i]

def test_sample_sizes_3():
    samples = 20
    num_inputs = 5
    limit = 23
    correct_sequence = [4, 4, 4, 4, 4]
    generated_sequence = list(_sample_sizes(samples, num_inputs, limit))
    assert len(correct_sequence) == len(generated_sequence)
    for i in range(len(correct_sequence)):
        assert correct_sequence[i] == generated_sequence[i]

def test_sample_sizes_4():
    samples = 20
    num_inputs = 5
    limit = 100
    correct_sequence = [20]
    generated_sequence = list(_sample_sizes(samples, num_inputs, limit))
    assert len(correct_sequence) == len(generated_sequence)
    for i in range(len(correct_sequence)):
        assert correct_sequence[i] == generated_sequence[i]

def test_sample_sizes_5():
    samples = 20
    num_inputs = 5
    limit = 101
    correct_sequence = [20]
    generated_sequence = list(_sample_sizes(samples, num_inputs, limit))
    assert len(correct_sequence) == len(generated_sequence)
    for i in range(len(correct_sequence)):
        assert correct_sequence[i] == generated_sequence[i]

def test_sample_sizes_6():
    samples = 20
    num_inputs = 5
    limit = 99
    correct_sequence = [19, 1]
    generated_sequence = list(_sample_sizes(samples, num_inputs, limit))
    assert len(correct_sequence) == len(generated_sequence)
    for i in range(len(correct_sequence)):
        assert correct_sequence[i] == generated_sequence[i]

def test_sample_sizes_7():
    samples = 50
    num_inputs = 5
    limit = 99
    correct_sequence = [19, 19, 12]
    generated_sequence = list(_sample_sizes(samples, num_inputs, limit))
    assert len(correct_sequence) == len(generated_sequence)
    for i in range(len(correct_sequence)):
        assert correct_sequence[i] == generated_sequence[i]

def test_sample_sizes_8():
    samples = 10
    num_inputs = 5
    limit = 100
    correct_sequence = [10]
    generated_sequence = list(_sample_sizes(samples, num_inputs, limit))
    assert len(correct_sequence) == len(generated_sequence)
    for i in range(len(correct_sequence)):
        assert correct_sequence[i] == generated_sequence[i]



#######################################
#        _sample_sizes tests          #
#######################################
def test_indices_and_batch_sizes_0():
    samples = 10
    batch_size = 3
    correct_sequence = [((0, 3), 3), ((3, 6), 3), ((6, 9), 3), ((9, 10), 1)]
    generated_sequence = list(_indices_and_batch_sizes(samples, batch_size))
    assert len(correct_sequence) == len(generated_sequence)
    for i in range(len(correct_sequence)):
        (ci0, ci1), csize = correct_sequence[i]
        (gi0, gi1), gsize = generated_sequence[i]
        assert ci0 == gi0
        assert ci1 == gi1
        assert csize == gsize

def test_indices_and_batch_sizes_1():
    samples = 10
    batch_size = 5
    correct_sequence = [((0, 5), 5), ((5, 10), 5)]
    generated_sequence = list(_indices_and_batch_sizes(samples, batch_size))
    assert len(correct_sequence) == len(generated_sequence)
    for i in range(len(correct_sequence)):
        (ci0, ci1), csize = correct_sequence[i]
        (gi0, gi1), gsize = generated_sequence[i]
        assert ci0 == gi0
        assert ci1 == gi1
        assert csize == gsize

def test_indices_and_batch_sizes_2():
    samples = 10
    batch_size = 15
    correct_sequence = [((0, 10), 10)]
    generated_sequence = list(_indices_and_batch_sizes(samples, batch_size))
    assert len(correct_sequence) == len(generated_sequence)
    for i in range(len(correct_sequence)):
        (ci0, ci1), csize = correct_sequence[i]
        (gi0, gi1), gsize = generated_sequence[i]
        assert ci0 == gi0
        assert ci1 == gi1
        assert csize == gsize



