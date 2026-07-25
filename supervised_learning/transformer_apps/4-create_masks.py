#!/usr/bin/env python3
"""
Utilities for building the masks required by the Transformer model.
"""

import tensorflow as tf


def create_masks(inputs, target):
    """
    Generate the encoder, decoder, and look-ahead masks.

    Args:
        inputs (tf.Tensor): Input tensor with shape
            (batch_size, seq_len_in).
        target (tf.Tensor): Target tensor with shape
            (batch_size, seq_len_out).

    Returns:
        tuple:
            encoder_mask,
            combined_mask,
            decoder_mask
    """

    def padding_mask(sequence):
        """
        Build a padding mask from a sequence.

        Padding tokens (0) are marked with 1.
        """
        mask = tf.cast(tf.equal(sequence, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    # Padding masks for encoder and cross-attention
    encoder_mask = padding_mask(inputs)
    decoder_mask = padding_mask(inputs)

    # Padding mask for decoder input
    target_mask = padding_mask(target)

    # Create look-ahead mask
    target_length = tf.shape(target)[1]
    future_mask = 1 - tf.linalg.band_part(
        tf.ones((target_length, target_length)),
        -1,
        0
    )

    # Merge padding and look-ahead masks
    combined_mask = tf.maximum(target_mask, future_mask)

    return encoder_mask, combined_mask, decoder_mask
