#!/usr/bin/env python3
"""TF-IDF embedding module"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix

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
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)

    embeddings = X.toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features
