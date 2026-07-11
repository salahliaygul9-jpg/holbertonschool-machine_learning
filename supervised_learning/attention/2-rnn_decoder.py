#!/usr/bin/env python3
"""RNN Decoder module"""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decoder for machine translation."""

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor.

        Args:
            vocab: size of the output vocabulary
            embedding: embedding dimension
            units: number of GRU hidden units
            batch: batch size
        """
        super(RNNDecoder, self).__init__()

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

        self.F = tf.keras.layers.Dense(vocab)

        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Forward pass.

        Args:
            x: previous target word, shape (batch, 1)
            s_prev: previous decoder hidden state,
                    shape (batch, units)
            hidden_states: encoder outputs,
                           shape (batch, input_seq_len, units)

        Returns:
            y: output logits, shape (batch, vocab)
            s: new decoder hidden state, shape (batch, units)
        """
        # Compute attention
        context, _ = self.attention(s_prev, hidden_states)

        # Embed input token
        x = self.embedding(x)

        # Expand context to concatenate with embedding
        context = tf.expand_dims(context, axis=1)

        # Concatenate context then embedding
        x = tf.concat([context, x], axis=-1)

        # Pass through GRU
        output, s = self.gru(
            x,
            initial_state=s_prev
        )

        # Remove sequence dimension
        output = tf.reshape(output, (-1, output.shape[2]))

        # Final dense layer
        y = self.F(output)

        return y, s
