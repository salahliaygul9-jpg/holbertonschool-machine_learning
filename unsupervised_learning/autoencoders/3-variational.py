#!/usr/bin/env python3
"""
Variational autoencoder.
"""

import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): input dimensions
        hidden_layers (list): hidden layer sizes
        latent_dims (int): latent space dimensions

    Returns:
        encoder, decoder, auto
    """

    # ====================
    # Encoder
    # ====================
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(
            nodes,
            activation='relu'
        )(x)

    mean = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    log_var = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    def sample(args):
        """
        Reparameterization trick.
        """
        mu, log_sigma = args

        epsilon = K.random_normal(
            shape=(K.shape(mu)[0], latent_dims)
        )

        return mu + K.exp(log_sigma / 2) * epsilon

    z = keras.layers.Lambda(
        sample
    )([mean, log_var])

    encoder = keras.Model(
        encoder_input,
        [z, mean, log_var]
    )

    # ====================
    # Decoder
    # ====================
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(
            nodes,
            activation='relu'
        )(x)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        decoder_input,
        decoder_output
    )

    # ====================
    # Autoencoder
    # ====================
    z, mean, log_var = encoder(encoder_input)
    outputs = decoder(z)

    auto = keras.Model(
        encoder_input,
        outputs
    )

    # KL Divergence
    kl_loss = -0.5 * K.sum(
        1 + log_var
        - K.square(mean)
        - K.exp(log_var),
        axis=1
    )

    auto.add_loss(K.mean(kl_loss))

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
