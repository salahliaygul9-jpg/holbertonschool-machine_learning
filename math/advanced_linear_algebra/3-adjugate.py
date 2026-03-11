#!/usr/bin/env python3
""" 3. Adjugate """
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """ calculates the adjugate matrix of a matrix """
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = []

    for i in range(len(cofactor_matrix[0])):
        row = []
        for j in range(len(cofactor_matrix)):
            row.append(cofactor_matrix[j][i])
        adjugate_matrix.append(row)

    return adjugate_matrix
