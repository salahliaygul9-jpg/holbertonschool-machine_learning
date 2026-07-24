#!/usr/bin/env python3
"""
Dataset class for machine translation.
"""

from setup import load_pt2en
import tensorflow_datasets as tfds
import transformers
Dataset class for Portuguese-to-English machine translation.
"""

import transformers
from setup import load_pt2en


class Dataset:
    """
    Dataset class that loads and prepares the Portuguese-English dataset.
    Loads and prepares the Portuguese-to-English dataset.
    """

    def __init__(self):
        """
        Class constructor.
        Initialize datasets and tokenizers.
        """
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates subword tokenizers for the dataset.

        Args:
            data: tf.data.Dataset containing (pt, en) sentence pairs.
        Creates Portuguese and English tokenizers.

        Args:
            data: tf.data.Dataset containing (pt, en) pairs.

        Returns:
            tokenizer_pt: Portuguese tokenizer.
            tokenizer_en: English tokenizer.
        """

        def pt_generator():
            """Yields Portuguese sentences."""
            for pt, _ in tfds.as_numpy(data):
                yield pt.decode("utf-8")

        def en_generator():
            """Yields English sentences."""
            for _, en in tfds.as_numpy(data):
                yield en.decode("utf-8")

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )

        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        vocab_size = 2 ** 13

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_generator(),
            vocab_size=vocab_size
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_generator(),
            vocab_size=vocab_size
        )

        del data

        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )

        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            "bert-base-uncased"
        )

        return tokenizer_pt, tokenizer_en
