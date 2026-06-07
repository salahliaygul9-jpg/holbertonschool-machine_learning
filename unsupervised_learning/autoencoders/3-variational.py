#!/usr/bin/env python3
"""
Variational autoencoder.
"""

import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.
    """

    # =================
    # Encoder
    # =================
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for h in hidden_layers:
        x = keras.layers.Dense(h, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims)(x)
    z_log_var = keras.layers.Dense(latent_dims)(x)

    def sample(args):
        mean, log_var = args
        eps = K.random_normal(shape=(K.shape(mean)[0], latent_dims))
        return mean + K.exp(log_var / 2) * eps

    z = keras.layers.Lambda(sample)([z_mean, z_log_var])

    encoder = keras.Model(
        encoder_input,
        [z, z_mean, z_log_var]
    )

    # =================
    # Decoder
    # =================
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for h in reversed(hidden_layers):
        x = keras.layers.Dense(h, activation='relu')(x)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(decoder_input, decoder_output)

    # =================
    # Autoencoder
    # =================
    z_out, mean_out, log_out = encoder(encoder_input)
    output = decoder(z_out)

    auto = keras.Model(encoder_input, output)

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
