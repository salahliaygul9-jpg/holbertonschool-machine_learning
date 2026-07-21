#!/usr/bin/env python3
"""
Dataset class for Portuguese-to-English machine translation.
"""

import transformers
from setup import load_pt2en


class Dataset:
    """
    Loads and prepares the Portuguese-to-English dataset.
    """

    def __init__(self):
        """
        Initialize datasets and tokenizers.
        """
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates Portuguese and English tokenizers.

        Args:
            data: tf.data.Dataset containing (pt, en) pairs.

        Returns:
            tokenizer_pt: Portuguese tokenizer.
            tokenizer_en: English tokenizer.
        """
        del data

        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )

        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            "bert-base-uncased"
        )

        return tokenizer_pt, tokenizer_en
