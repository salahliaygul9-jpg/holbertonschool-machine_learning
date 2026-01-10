#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file

df = from_file('test.csv', ',')
print(df)
