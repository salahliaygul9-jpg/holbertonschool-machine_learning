#!/usr/bin/env python3
"""Self Attention module"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Bahdanau Self-Attention"""

    def __init__(self, units):
        """
        Class constructor

        Args:
            units: number of hidden units in the alignment model
        """
        super(SelfAttention, self).__init__()

        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Performs the forward pass for attention.

        Args:
            s_prev: previous decoder hidden state,
                    shape (batch, units)
            hidden_states: encoder outputs,
                           shape (batch, input_seq_len, units)

        Returns:
            context: context vector, shape (batch, units)
            weights: attention weights,
                     shape (batch, input_seq_len, 1)
        """
        # Expand decoder hidden state for broadcasting
        s_prev = tf.expand_dims(s_prev, axis=1)

        # Alignment scores
        score = self.V(
            tf.nn.tanh(
                self.W(s_prev) + self.U(hidden_states)
            )
        )

        # Attention weights
        weights = tf.nn.softmax(score, axis=1)

        # Context vector
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, weights
