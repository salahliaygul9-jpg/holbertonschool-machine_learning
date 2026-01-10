#!/usr/bin/env python3

'''
This module contains the high function
'''

def high(df):
    """
    This function replaces NaN values in High column with Close values
    """
    df = df.drop('Weighted_Price', axis=1)
    df['Close'] = df['Close'].fillna(method='pad')
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    return df
