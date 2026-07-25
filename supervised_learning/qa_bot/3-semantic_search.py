#!/usr/bin/env python3
"""Semantic search on a corpus of reference documents."""
 
import os
import numpy as np
import tensorflow_hub as hub
 
 
def semantic_search(corpus_path, sentence):
    """Perform semantic search on a corpus of documents.
 
    Args:
        corpus_path (str): path to the corpus of reference documents
        sentence (str): sentence from which to perform semantic search
 
    Returns:
        str: the reference text of the document most similar to sentence
    """
    model = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5'
    )
 
    documents = [sentence]
    filenames = []
 
    for filename in os.listdir(corpus_path):
        if not filename.endswith('.md'):
            continue
 
        filepath = os.path.join(corpus_path, filename)
 
        with open(filepath, 'r', encoding='utf-8') as f:
            documents.append(f.read())
 
        filenames.append(filename)
 
    embeddings = model(documents)
 
    sentence_embedding = embeddings[0]
    document_embeddings = embeddings[1:]
 
    correlation = np.inner(sentence_embedding, document_embeddings)
    closest = np.argmax(correlation)
 
    closest_filepath = os.path.join(corpus_path, filenames[closest])
 
    with open(closest_filepath, 'r', encoding='utf-8') as f:
        return f.read()
 
