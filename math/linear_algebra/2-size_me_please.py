#!/usr/bin/env python3
'''Size me up function'''


def matrix_shape(matrix):
    '''Comment for matrix shape function'''
    result = []
    while (type(matrix) is list):
        result.append(len(matrix))
        matrix = matrix[0]
    return result


mat1 = [[1, 2], [3, 4]]
matrix_shape(mat1)
mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
matrix_shape(mat2)
