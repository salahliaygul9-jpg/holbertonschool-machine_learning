#!/usr/bin/env python3
"""Convolutional Back Prop"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """performs back propagation over a convolutional layer of
    a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    m, h_new, w_new, c_new = dZ.shape
    dW = np.zeros_like(W)
    da = np.zeros_like(A_prev)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    if padding == 'valid':
        p_h, p_w = 0, 0
    elif padding == 'same':
        p_h = (((h_prev - 1) * sh) + kh - h_prev) // 2 + 1
        p_w = (((w_prev - 1) * sw) + kw - w_prev) // 2 + 1
    A_prev = np.pad(A_prev, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)),
                    mode='constant', constant_values=0)
    dA = np.pad(da, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)),
                mode='constant', constant_values=0)
    for frame in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for k in range(c_new):
                    kernel = W[:, :, :, k]
                    dz = dZ[frame, h, w, k]
                    mat = A_prev[frame, sh*h:sh*h+kh, sw*w:sw*w+kw, :]
                    dW[:, :, :, k] += mat * dz
                    dA[frame, sh*h:sh*h+kh, sw*w:sw*w+kw,
                       :] += np.multiply(kernel, dz)
    if padding == 'same':
        dA = dA[:, p_h: -p_h, p_w: -p_w, :]
    return dA, dW, db
