#!/usr/bin/env python3
"""
Variational Autoencoder
"""

import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): number of nodes for each hidden layer
        latent_dims (int): dimensions of the latent space representation

    Returns:
        encoder, decoder, auto
    """

    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    for units in hidden_layers:
        x = keras.layers.Dense(
            units,
            activation='relu'
        )(x)

    z_mean = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    z_log_var = keras.layers.Dense(
        latent_dims,
        activation=None
    )(x)

    def sampling(args):
        """Reparameterization trick."""
        mean, log_var = args

        epsilon = K.random_normal(
            shape=(K.shape(mean)[0], latent_dims)
        )

        return mean + K.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(
        sampling
    )([z_mean, z_log_var])

    encoder = keras.Model(
        encoder_inputs,
        [z, z_mean, z_log_var]
    )

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs

    for units in reversed(hidden_layers):
        x = keras.layers.Dense(
            units,
            activation='relu'
        )(x)

    decoder_outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        decoder_inputs,
        decoder_outputs
    )

    # Autoencoder
    outputs = decoder(z)

    auto = keras.Model(
        encoder_inputs,
        outputs
    )

    # KL divergence loss
    kl_loss = -0.5 * K.sum(
        1 + z_log_var
        - K.square(z_mean)
        - K.exp(z_log_var),
        axis=-1
    )

    auto.add_loss(K.mean(kl_loss))

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
