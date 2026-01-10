#!/usr/bin/env python3

'''
This module selects every 60th row of these columns
'''


def slice(df):
    '''
    This fumction does same thing like above
    '''
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
