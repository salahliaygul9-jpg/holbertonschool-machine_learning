#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images (numpy.ndarray): shape (m, h, w) containing multiple grayscale images
            - m: number of images
            - h: height of each image
            - w: width of each image
        kernel (numpy.ndarray): shape (kh, kw) containing the kernel
            - kh: height of the kernel
            - kw: width of the kernel

    Returns:
        numpy.ndarray: the convolved images with shape (m, h - kh + 1, w - kw + 1)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Output dimensions for valid convolution
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize output array
    convolved = np.zeros((m, output_h, output_w))

    # Only two for loops allowed: one for images, one for output height
    for i in range(m):
        for y in range(output_h):
            # Use slicing for the width dimension (no third loop)
            for x in range(output_w):
                # Extract the region and perform convolution (dot product)
                region = images[i, y:y+kh, x:x+kw]
                convolved[i, y, x] = np.sum(region * kernel)

    return convolved
