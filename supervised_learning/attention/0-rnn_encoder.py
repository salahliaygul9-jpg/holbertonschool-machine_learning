#!/usr/bin/env python3
"""RNN Encoder for machine translation"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encoder class"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        Args:
            vocab: size of the input vocabulary
            embedding: dimensionality of the embedding vectors
            units: number of GRU hidden units
            batch: batch size
        """
        super(RNNEncoder, self).__init__()

        self.batch = batch
        self.units = units

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )

    def initialize_hidden_state(self):
        """
        Initializes the hidden state

        Returns:
            Tensor of zeros with shape (batch, units)
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Forward pass

        Args:
            x: input tensor of shape (batch, input_seq_len)
            initial: initial hidden state

        Returns:
            outputs: encoder outputs
            hidden: final hidden state
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(
            x,
            initial_state=initial
        )

        return outputs, hidden
