#!/usr/bin/env python3
"""
Unigram BLEU score
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    references is a list of reference translations
    each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    Returns: the unigram BLEU score
    """

    len_sentence = len(sentence)
    len_reference = []
    dict_words = {}

    for i in references:
        len_reference.append(len(i))

        for word in i:
            if word in sentence:
                if not dict_words.keys() == word:
                    dict_words[word] = 1

    prob = sum(dict_words.values())
    ind = np.argmin([abs(len(x) - len_sentence) for x in references])
    best_match = len(references[ind])

    if len_sentence > best_match:
        bp = 1
    else:
        bp = np.exp(1 - float(best_match) / float(len_sentence))

    Blue_score = bp * np.exp(np.log(prob / len_sentence))

    return Blue_score
