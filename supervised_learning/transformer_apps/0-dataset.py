#!/usr/bin/env python3
"""
Load, tokenize tensorflow Dataset
"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    A class to load and prepare the TED HRLR translation dataset
    for machine translation from Portuguese to English.
    """

    def __init__(self):
        """
        Initializes the Dataset object and loads the training and
        validation datasets. Also initializes tokenizers for
        Portuguese and English.
        """
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True,
            try_gcs=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True,
            try_gcs=True
        )

        # Initialize tokenizers
"""Dataset class for Portuguese-to-English machine translation."""

import transformers
from setup import load_pt2en


class Dataset:
    """Loads and prepares the Portuguese-to-English dataset."""

    def __init__(self):
        """Initialize datasets and tokenizers."""
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset using pre-trained
        models.

        Args:
            data: tf.data.Dataset containing tuples of (pt, en)
            sentences.

        Returns:
            tokenizer_pt: Tokenizer for Portuguese.
            tokenizer_en: Tokenizer for English.
        """
        # Load the pre-trained tokenizers directly
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        """Create Portuguese and English subword tokenizers."""
        base_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        base_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        def pt_iterator():
            for pt, _ in data:
                yield pt.numpy().decode("utf-8")

        def en_iterator():
            for _, en in data:
                yield en.numpy().decode("utf-8")

        tokenizer_pt = base_pt.train_new_from_iterator(
            pt_iterator(),
            vocab_size=2 ** 13,
        )

        tokenizer_en = base_en.train_new_from_iterator(
            en_iterator(),
            vocab_size=2 ** 13,
        )

        return tokenizer_pt, tokenizer_en
