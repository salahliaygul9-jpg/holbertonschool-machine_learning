#!/usr/bin/env python3
"""
Dataset class for Portuguese-English translation
"""

from setup import load_pt2en
from transformers import BertTokenizerFast


class Dataset:
    """Loads and prepares the translation dataset"""

    def __init__(self):
        """Class constructor"""

        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = (
            self.tokenize_dataset(self.data_train)
        )

    def tokenize_dataset(self, data):
        """
        Creates tokenizers for the dataset

        Args:
            data: tf.data.Dataset of (pt, en) sentence pairs

        Returns:
            tokenizer_pt, tokenizer_en
        """

        tokenizer_pt = BertTokenizerFast.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )

        tokenizer_en = BertTokenizerFast.from_pretrained(
            "bert-base-uncased"
        )

        return tokenizer_pt, tokenizer_en
