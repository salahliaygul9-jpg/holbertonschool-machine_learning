#!/usr/bin/env python3
"""
Convolutional autoencoder.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): dimensions of the model input
        filters (list): number of filters for each encoder layer
        latent_dims (tuple): dimensions of the latent space

    Returns:
        encoder, decoder, auto
    """

    # Encoder
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input

    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)

        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(x)

    encoder_output = x

    encoder = keras.Model(
        encoder_input,
        encoder_output
    )

    # Decoder
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input

    reversed_filters = filters[::-1]

    for f in reversed_filters[:-1]:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)

        x = keras.layers.UpSampling2D(
            size=(2, 2)
        )(x)

    x = keras.layers.Conv2D(
        filters=reversed_filters[-1],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(x)

    x = keras.layers.UpSampling2D(
        size=(2, 2)
    )(x)

    decoder_output = keras.layers.Conv2D(
        filters=input_dims[2],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        decoder_input,
        decoder_output
    )

    # Autoencoder
    auto_input = encoder_input
    encoded = encoder(auto_input)
    decoded = decoder(encoded)

    auto = keras.Model(
        auto_input,
        decoded
    )

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
