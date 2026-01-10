#!/usr/bin/env python3
"""
This module loads data from a file as a DataFrame using delimiter
"""
import pandas as pd

def from_file(filename, delimiter):
    """
    Loads data from a CSV (or delimited) file into a pandas DataFrame
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
