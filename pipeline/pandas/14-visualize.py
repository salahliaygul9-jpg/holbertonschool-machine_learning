#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file


def visualize():
    """
    A function that visualizes and returns the transformed DataFrame
    """
    df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
    df = df.drop(columns=['Weighted_Price'])
    df = df.rename(columns={'Timestamp': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], unit='s')
    df = df.set_index('Date')
    df['Close'] = df['Close'].ffill()
    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])
    for col in ['Volume_(BTC)', 'Volume_(Currency)']:
        df[col] = df[col].fillna(0)
    df = df[df.index.year >= 2017]
    daily_df = df.resample('D').agg({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })
    print(daily_df)
    daily_df.plot(figsize=(12, 6))
    plt.title('Coinbase BTC USD Daily Data (2017 onwards)')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return daily_df


if __name__ == "__main__":
    df_daily = visualize()
