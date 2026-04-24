#!/usr/bin/env python3
"""
12-learning_rate_decay.py
Creates a TensorFlow inverse time decay learning rate schedule
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in TensorFlow
    using inverse time decay (stepwise)

    alpha: original learning rate
    decay_rate: decay rate
    decay_step: number of steps before further decay
    Returns: a TensorFlow learning rate schedule
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
