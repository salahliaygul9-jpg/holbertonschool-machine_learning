#!/usr/bin/env python3

"""
This module fills missing OHLC and Volume data
"""

def high(df):
    """
    Fill NaNs in OHLC and Volume columns
    """
    # fill Close forward
    df['Close'] = df['Close'].ffill()
    # fill High, Low, Open using Close
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])
    # fill Volume columns with 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df

