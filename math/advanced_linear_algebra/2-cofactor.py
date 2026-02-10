#!/usr/bin/env python3
"""
Module to calculate the cofactor matrix of a matrix
"""

determinant = __import__('0-determinant').determinant
minor = __import__('1-minor').minor


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix

    Args:
        matrix (list of lists): Matrix
            whose cofactor matrix should be calculated

    Returns:
        list of lists: The cofactor matrix of matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square or is empty
    """
    # Check if matrix is a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is empty
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if matrix is square
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Get the minor matrix
    minor_matrix = minor(matrix)

    # Calculate the cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            # Apply the checkerboard pattern of signs
            cofactor_row.append(minor_matrix[i][j] * ((-1) ** (i + j)))
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
