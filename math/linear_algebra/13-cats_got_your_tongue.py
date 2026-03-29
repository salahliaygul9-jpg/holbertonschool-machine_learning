#!/usr/bin/env python3
"""function concatenate that concatenates two matrices along a specific axis"""
import numpy as np
"""function concatenate that concatenates two matrices along a specific axis"""


def np_cat(mat1, mat2, axis=0):
    """function concate that concatenates two matrices along a specific axis"""
    mat = np.concatenate((mat1, mat2), axis)
    return mat
