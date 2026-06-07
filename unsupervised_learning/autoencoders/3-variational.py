#!/usr/bin/env python3
"""
Variational autoencoder.
"""

import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): input size
        hidden_layers (list): encoder hidden layers
        latent_dims (int): latent dimension

    Returns:
        encoder, decoder, auto
    """

    # =====================
    # Encoder
    # =====================
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    for h in hidden_layers:
        x = keras.layers.Dense(h, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims)(x)
    z_log_var = keras.layers.Dense(latent_dims)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dims))
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z, z_mean, z_log_var])

    # =====================
    # Decoder
    # =====================
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    for h in reversed(hidden_layers):
        x = keras.layers.Dense(h, activation='relu')(x)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_inputs, outputs)

    # =====================
    # Autoencoder
    # =====================
    z_out, mean_out, log_out = encoder(inputs)
    recon = decoder(z_out)

    auto = keras.Model(inputs, recon)

    # KL loss
    kl = -0.5 * K.sum(
        1 + log_out - K.square(mean_out) - K.exp(log_out),
        axis=-1
    )

    auto.add_loss(K.mean(kl))

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
