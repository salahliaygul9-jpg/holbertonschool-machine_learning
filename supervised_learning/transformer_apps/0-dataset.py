#!/usr/bin/env python3
"""Defines the Dataset class used to load and prep a machine
translation dataset for Portuguese to English translation.
"""
from setup import load_pt2en
from transformers import AutoTokenizer


class Dataset:
    """Loads and preps a dataset for machine translation."""

    def __init__(self):
        """Class constructor.

        Sets the following public instance attributes:
            data_train: the ted_hrlr_translate/pt_to_en train split,
                loaded via load_pt2en('train')
            data_valid: the ted_hrlr_translate/pt_to_en validation
                split, loaded via load_pt2en('validation')
            tokenizer_pt: the Portuguese tokenizer created from the
                training set
            tokenizer_en: the English tokenizer created from the
                training set
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for the dataset.

        Args:
            data: a tf.data.Dataset whose examples are formatted as
                a tuple (pt, en):
                    pt is the tf.Tensor containing the Portuguese
                        sentence
                    en is the tf.Tensor containing the corresponding
                        English sentence

        Returns:
            tokenizer_pt, tokenizer_en
                tokenizer_pt is the Portuguese tokenizer
                tokenizer_en is the English tokenizer
        """
        pt_base = AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        en_base = AutoTokenizer.from_pretrained('bert-base-uncased')

        def pt_sentences():
            """Yields decoded Portuguese sentences from data."""
            for pt, _ in data.as_numpy_iterator():
                yield pt.decode('utf-8')

        def en_sentences():
            """Yields decoded English sentences from data."""
            for _, en in data.as_numpy_iterator():
                yield en.decode('utf-8')

        tokenizer_pt = pt_base.train_new_from_iterator(
            pt_sentences(), vocab_size=2 ** 13)
        tokenizer_en = en_base.train_new_from_iterator(
            en_sentences(), vocab_size=2 ** 13)

        return tokenizer_pt, tokenizer_en
