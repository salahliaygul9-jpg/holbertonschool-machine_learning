#!/usr/bin/env python3
"""
Module contains function for creating and training
a gensim word2vec model.
"""


from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0,
                   workers=1):

    """
    Creates and trains a gensim word2vec model.

    Args:
        sentences: List of sentences to be trained on.
        size: Dimensionality of the embedding layer.
        min_count: Minimum number of occurrences of a
        word for use in training.
        window: Maximum distance between the current and
        predicted word within a sentence.
        negative: Size of negative sampling.
        cbow: Boolean to determine the training type;
        True is for CBOW; False is for Skip-gram.
        iterations: Number of iterations to train over.
        seed: Seed for the random number generator.
        workers: Number of worker threads to train the model.

    Return: The trained model.
    """

    sg = 0 if cbow else 1

    model = Word2Vec(
        sentences=sentences, sg=sg, negative=negative,
        window=window, min_count=min_count, workers=workers,
        seed=seed, size=size)

    model.train(
        sentences, epochs=iterations,
        total_examples=model.corpus_count)
Creates and trains a gensim word2vec model
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a gensim word2vec model

    Args:
        sentences (list): A list of sentences to be trained on
        size (int): The dimensionality of the embedding layer
        min_count (int): The minimum number of occurrences of a
        word for use in training
        window (int): The maximum distance between the current
        and predicted word within a sentence
        negative (int): The size of negative sampling
        cbow (bool): Boolean to determine the training type; True is for CBOW;
        False is for Skip-gram
        iterations (int): The number of iterations to train the model
        seed (int): The seed for the random number generator
        workers (int): The number of worker threads to train the model

    Returns:
        Word2Vec: The trained word2vec model
    """
    sg = 0 if cbow else 1  # Set sg to 0 for CBOW, 1 for Skip-gram

    model = Word2Vec(size=size, window=window,
                     min_count=min_count, workers=workers,
                     sg=sg, negative=negative, seed=seed)
    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=iterations)

    return model
