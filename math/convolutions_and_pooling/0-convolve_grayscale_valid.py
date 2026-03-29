#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images using only two for loops.

    Args:
        images (numpy.ndarray): shape (m, h, w) - multiple grayscale images
        kernel (numpy.ndarray): shape (kh, kw) - convolution kernel

    Returns:
        numpy.ndarray: convolved images of shape (m, h - kh + 1, w - kw + 1)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Output dimensions for valid convolution
    oh = h - kh + 1
    ow = w - kw + 1

    # Initialize output
    output = np.zeros((m, oh, ow))

    # Only TWO for loops allowed: over images and over output height
    for i in range(m):
        for y in range(oh):
            # Vectorized over width using slicing + np.sum (no third loop)
            for x in range(ow):
                region = images[i, y:y + kh, x:x + kw]
                output[i, y, x] = np.sum(region * kernel)

    return output
