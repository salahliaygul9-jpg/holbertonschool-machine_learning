#!/usr/bin/env python3
"""
Dataset class for loading, tokenizing, and encoding the
Portuguese-English translation dataset.
"""

import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Dataset wrapper used for machine translation."""

    def __init__(self):
        """Initialize datasets, tokenizers, and encoded datasets."""
        self.data_train = load_pt2en(split="train")
        self.data_valid = load_pt2en(split="validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        self.data_train = self.data_train.map(
            lambda pt, en: self.tf_encode(pt, en)
        )
        self.data_valid = self.data_valid.map(
            lambda pt, en: self.tf_encode(pt, en)
        )

    def tokenize_dataset(self, data):
        """
        Train Portuguese and English tokenizers.

        Args:
            data: TensorFlow dataset.

        Returns:
            tuple containing both trained tokenizers.
        """
        pt_tokenizer = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        en_tokenizer = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        def get_portuguese():
            """Yield Portuguese sentences."""
            for pt, _ in data.batch(1000).as_numpy_iterator():
                yield [text.decode("utf-8") for text in pt]

        def get_english():
            """Yield English sentences."""
            for _, en in data.batch(1000).as_numpy_iterator():
                yield [text.decode("utf-8") for text in en]

        vocab = 2 ** 13

        tokenizer_pt = pt_tokenizer.train_new_from_iterator(
            get_portuguese(),
            vocab
        )
        tokenizer_en = en_tokenizer.train_new_from_iterator(
            get_english(),
            vocab
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encode Portuguese and English sentences.

        Args:
            pt: Portuguese sentence tensor.
            en: English sentence tensor.

        Returns:
            Tuple containing encoded Portuguese and English sequences.
        """
        start = self.tokenizer_pt.vocab_size
        end = start + 1

        pt_sentence = pt.numpy().decode("utf-8")
        en_sentence = en.numpy().decode("utf-8")

        pt_tokens = self.tokenizer_pt.encode(
            pt_sentence,
            add_special_tokens=False
        )
        en_tokens = self.tokenizer_en.encode(
            en_sentence,
            add_special_tokens=False
        )

        pt_result = [start] + pt_tokens + [end]
        en_result = [start] + en_tokens + [end]

        return pt_result, en_result

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper around encode().

        Args:
            pt: Portuguese sentence tensor.
            en: English sentence tensor.

        Returns:
            Encoded Portuguese and English tensors.
        """
        pt_tensor, en_tensor = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )

        pt_tensor.set_shape([None])
        en_tensor.set_shape([None])

        return pt_tensor, en_tensor
