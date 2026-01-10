#!/usr/bin/env python3
"""
rename the data frame
"""

import pandas as pd


def rename(df):
    """
    inside the function
    """
    modified_df = df.rename(columns={"Timestamp": "Datetime"})
    modified_df["Datetime"] = pd.to_datetime(modified_df["Datetime"], unit="s")
    return modified_df[["Datetime", "Close"]]
