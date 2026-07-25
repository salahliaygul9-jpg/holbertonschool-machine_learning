#!/usr/bin/env python3
"""
Dataset module for preparing the Portuguese-English
translation dataset used by the transformer model.
"""

import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Loads, tokenizes, encodes, and prepares the dataset."""

    def __init__(self, batch_size, max_len):
        """
        Build the dataset pipeline.

        Args:
            batch_size (int): Number of examples per batch.
            max_len (int): Maximum allowed sequence length.
        """
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

        self.data_train = (
            self.data_train
            .filter(lambda pt, en: self._valid_length(pt, en, max_len))
            .cache()
            .shuffle(buffer_size=20000)
            .padded_batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        self.data_valid = (
            self.data_valid
            .filter(lambda pt, en: self._valid_length(pt, en, max_len))
            .padded_batch(batch_size)
        )

    def _valid_length(self, pt, en, max_len):
        """
        Check whether both encoded sentences satisfy
        the maximum sequence length.
        """
        return tf.logical_and(
            tf.size(pt) <= max_len,
            tf.size(en) <= max_len
        )

    def tokenize_dataset(self, data):
        """
        Train Portuguese and English tokenizers.

        Args:
            data: TensorFlow dataset.

        Returns:
            tuple: Portuguese and English tokenizers.
        """
        pt_base = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        en_base = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        def portuguese_sentences():
            """Yield Portuguese sentences."""
            for pt, _ in data.batch(1000).as_numpy_iterator():
                yield [text.decode("utf-8") for text in pt]

        def english_sentences():
            """Yield English sentences."""
            for _, en in data.batch(1000).as_numpy_iterator():
                yield [text.decode("utf-8") for text in en]

        vocab_size = 2 ** 13

        tokenizer_pt = pt_base.train_new_from_iterator(
            portuguese_sentences(),
            vocab_size
        )
        tokenizer_en = en_base.train_new_from_iterator(
            english_sentences(),
            vocab_size
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Convert a sentence pair into token IDs.

        Args:
            pt: Portuguese sentence tensor.
            en: English sentence tensor.

        Returns:
            Tuple containing encoded Portuguese
            and English token sequences.
        """
        start_token = self.tokenizer_pt.vocab_size
        end_token = start_token + 1

        pt_text = pt.numpy().decode("utf-8")
        en_text = en.numpy().decode("utf-8")

        pt_ids = self.tokenizer_pt.encode(
            pt_text,
            add_special_tokens=False
        )
        en_ids = self.tokenizer_en.encode(
            en_text,
            add_special_tokens=False
        )

        return (
            [start_token] + pt_ids + [end_token],
            [start_token] + en_ids + [end_token]
        )

    def tf_encode(self, pt, en):
        """
        Wrap encode() for use inside TensorFlow pipelines.

        Args:
            pt: Portuguese sentence tensor.
            en: English sentence tensor.

        Returns:
            Encoded TensorFlow tensors.
        """
        pt_tokens, en_tokens = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
