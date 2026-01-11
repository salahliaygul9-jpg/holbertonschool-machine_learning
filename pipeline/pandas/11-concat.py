#!/usr/bin/env python3
"""
asgsagsagsaa
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    asjkakjsjkasjks
    """
    df1 = index(df1)
    df2 = index(df2)

    df2f = df2[df2.index <= 1417411920]
    result = pd.concat(
        [df2f, df1],
        keys=['bitstamp', 'coinbase']
    )

    return result
