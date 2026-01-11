#!/usr/bin/env python3
"""
Module for Stacked Bar Chart
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plot stacked bar chart of fruit quantities per person"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    x = ['Farrah', 'Fred', 'Felicia']
    plt.bar(x, fruit[0], color='red', width=0.5, label='apples')
    plt.bar(x, fruit[1], bottom=fruit[0], color='yellow',
            width=0.5, label='bananas')
    plt.bar(x, fruit[2], bottom=fruit[0] + fruit[1],
            color="#ff8000", width=0.5, label='oranges')
    plt.bar(x, fruit[3], bottom=fruit[0] + fruit[1] + fruit[2],
            color="#ffe5b4", width=0.5, label='peaches')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.show()
