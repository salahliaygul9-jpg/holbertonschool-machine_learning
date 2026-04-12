#!/usr/bin/env python3
"""Pooling Forward Prop"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs forward propagation over a pooling layer of
    a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    if mode == 'max':
        op = np.max
    elif mode == 'avg':
        op = np.mean
    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1
    pool = np.zeros((m, h_out, w_out, c_prev))
    for h in range(h_out):
        for w in range(w_out):
            pool[:, h, w, :] = op(A_prev[:, sh*h:sh*h+kh, sw*w:sw*w+kw],
                                  axis=(1, 2))
    return pool
