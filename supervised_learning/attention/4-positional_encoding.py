#!/usr/bin/env python3
"""Positional Encoding"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer.

    Args:
        max_seq_len: maximum sequence length
        dm: model depth

    Returns:
        A numpy.ndarray of shape (max_seq_len, dm)
        containing the positional encoding vectors.
    """
    # Position indices: (max_seq_len, 1)
    positions = np.arange(max_seq_len)[:, np.newaxis]

    # Dimension indices: (1, dm)
    dimensions = np.arange(dm)[np.newaxis, :]

    # Angle rates
    angle_rates = positions / np.power(
        10000,
        (2 * (dimensions // 2)) / dm
    )

    # Initialize positional encoding matrix
    PE = np.zeros((max_seq_len, dm))

    # Apply sin to even indices
    PE[:, 0::2] = np.sin(angle_rates[:, 0::2])

    # Apply cos to odd indices
    PE[:, 1::2] = np.cos(angle_rates[:, 1::2])

    return PE
