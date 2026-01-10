#!/usr/bin/env python3

'''
This module flips data
'''


def flip_switch(df):
    '''
    This fumction does same thing like above
    '''
    return df.iloc[::-1].transpose()
