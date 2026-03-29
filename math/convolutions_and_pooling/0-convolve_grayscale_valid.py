#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images (numpy.ndarray): Input images of shape (m, h, w)
                                m: number of images
                                h: height of each image
                                w: width of each image
        kernel (numpy.ndarray): Kernel of shape (kh, kw)

    Returns:
        numpy.ndarray: Convolved images of shape (m, h - kh + 1, w - kw + 1)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate output dimensions (valid convolution)
    oh = h - kh + 1
    ow = w - kw + 1

    # Initialize the output array
    convolved = np.zeros((m, oh, ow))

    # Only two outer for loops (as required by the task)
    for i in range(m):
        for y in range(oh):
            for x in range(ow):
                # Element-wise multiplication + sum (very fast in numpy)
                region = images[i, y:y + kh, x:x + kw]
                convolved[i, y, x] = np.sum(region * kernel)

    return convolved
