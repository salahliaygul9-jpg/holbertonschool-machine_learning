#!/usr/bin/env python3
"""Bag of Words embedding module"""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use for the analysis
               If None, all words within sentences should be used

    Returns:
        embeddings, features
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
            s is the number of sentences in sentences
            f is the number of features analyzed
        features: list of the features used for embeddings
    """
    def tokenize(sentence):
        """Lowercase, strip possessive 's, and extract word tokens"""
        sentence = sentence.lower()
        # remove possessive 's (e.g. "children's" -> "children")
        sentence = re.sub(r"'s\b", "", sentence)
        return re.findall(r"[a-z]+", sentence)

    tokenized_sentences = [tokenize(sentence) for sentence in sentences]

    if vocab is None:
        vocab_set = set()
        for words in tokenized_sentences:
            vocab_set.update(words)
        features = sorted(vocab_set)
    else:
        features = list(vocab)

    feature_index = {word: idx for idx, word in enumerate(features)}

    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, words in enumerate(tokenized_sentences):
        for word in words:
            if word in feature_index:
                embeddings[i, feature_index[word]] += 1

    features = np.array(features)

    return embeddings, features
