#!/usr/bin/env python3

'''
This module loads data from a file as a DataFrame
by using delimiter
'''
import pandas as pd


def from_file(filename, delimiter):
    '''
    This function loads data with delimiter
    '''
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
