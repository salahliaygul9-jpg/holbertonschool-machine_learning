#!/usr/bin/env python3
"""
L2 Regularization Cost
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Computes the total cost of a neural network including
    L2 regularization.

    Args:
        cost: tensor representing the base cost without regularization

    Returns:
        Tensor representing the total cost with L2 regularization added
    """
    return cost + tf.add_n(tf.losses.get_regularization_losses())
