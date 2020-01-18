import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from ei import MI

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

