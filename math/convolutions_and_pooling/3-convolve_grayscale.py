#!/usr/bin/env python3
""" convolutions """
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ Performs a convolution on grayscale images with padding and stride.

    Args:
        images (numpy.ndarray): shape (m, h, w) containing multiple grayscale images
        kernel (numpy.ndarray): shape (kh, kw) containing the kernel
        padding (str or tuple): 'same', 'valid', or (ph, pw)
        stride (tuple): (sh, sw)

    Returns:
        numpy.ndarray: the convolved images
    """
    kh, kw = kernel.shape
    m, hm, wm = images.shape
    sh, sw = stride

    # Calculate padding
    if padding == 'same':
        ph = int(((hm - 1) * sh + kh - hm) / 2) + 1
        pw = int(((wm - 1) * sw + kw - wm) / 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    # Apply padding with zeros
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    # Calculate output dimensions
    ch = int((hm + 2 * ph - kh) / sh) + 1
    cw = int((wm + 2 * pw - kw) / sw) + 1

    # Initialize output array
    convoluted = np.zeros((m, ch, cw))

    # Only two for loops allowed
    for h in range(ch):
        for w in range(cw):
            # Extract region for all images at once
            square = padded[:, h * sh:h * sh + kh, w * sw:w * sw + kw]
            # Convolution: multiply and sum over kernel dimensions
            insert = np.sum(square * kernel, axis=(1, 2))
            convoluted[:, h, w] = insert

    return convoluted
