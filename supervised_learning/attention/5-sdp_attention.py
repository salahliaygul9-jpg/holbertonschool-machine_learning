#!/usr/bin/env python3
"""Scaled Dot-Product Attention"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot-product attention.

    Args:
        Q: query tensor (..., seq_len_q, dk)
        K: key tensor (..., seq_len_v, dk)
        V: value tensor (..., seq_len_v, dv)
        mask: optional mask broadcastable to
              (..., seq_len_q, seq_len_v)

    Returns:
        output: (..., seq_len_q, dv)
        weights: (..., seq_len_q, seq_len_v)
    """
    # QK^T
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale by sqrt(dk)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply mask if provided
    if mask is not None:
        scaled_logits += (mask * -1e9)

    # Softmax to obtain attention weights
    weights = tf.nn.softmax(scaled_logits, axis=-1)

    # Weighted sum of values
    output = tf.matmul(weights, V)

    return output, weights
