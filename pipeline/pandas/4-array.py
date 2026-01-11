#!/usr/bin/env python3
"""
to numpy
"""


def array(df):
    """
    inside the func
    """
    df = df.tail(10)
    df = df[['High', 'Close']]
    numpy_array = df.to_numpy()
    return numpy_array
