#!/usr/bin/env python
# coding: utf-8


import time

import numpy as np
import torch


def timeit(size, iterations=10, cuda=False):
    m = torch.tensor(np.random.randn(size, size))
    t0 = time.time()
    if cuda:
        m = m.to('cuda')
    for _ in range(iterations):
        m = torch.matmul(m, m)
    device = 'cpu' if not cuda else 'gpu'
    print('{:.4f}s for {} multiplies of {}x{} on {}'.format(time.time() - t0, iterations, size, size, device))


if __name__ == '__main__':
    timeit(1000, iterations=10, cuda=False)
    timeit(1000, iterations=10, cuda=True)
    timeit(5000, iterations=5, cuda=False)
    timeit(5000, iterations=5, cuda=True)
    timeit(10000, iterations=1, cuda=False)
    timeit(10000, iterations=1, cuda=True)


