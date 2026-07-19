#!/usr/bin/env python3
"""Module for the class Dataset"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """Loads and preps a dataset for machine translation.
    """
    def __init__(self):
        """Class constructor.
        """

        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)

        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)

        tokenizer_pt, tokenizer_en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for our dataset.

        Args.
            data: tf.data.Dataset whose examples are formatted as a tuple
                (pt, en).

        Returns.
            The Portuguese tokenizer and the English tokenizer.
        """

        bfc = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        tokenizer_pt = bfc((pt.numpy() for pt, en in data),
                           target_vocab_size=2**15)

        tokenizer_en = bfc((en.numpy() for pt, en in data),
                           target_vocab_size=2**15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encodes a translation into tokens.

        Args.
            pt: tf.Tensor containing the Portuguese sentence.
            en: tf.Tensor containing the corresponding English sentence.

        Returns.
            An np.ndarray containing the Portuguese tokens and an np.ndarray
            containing the English tokens
        """
        v_size = self.tokenizer_pt.vocab_size
        pt_tokens = [v_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [v_size + 1]
        v_size = self.tokenizer_en.vocab_size
        en_tokens = [v_size] + self.tokenizer_en.encode(
            en.numpy()) + [v_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Acts as a tensorflow wrapper for the encode instance method.

        Args.
            pt: tf.Tensor containing the Portuguese sentence.
            en: tf.Tensor containing the corresponding English sentence.

        Returns.
            A a tf.tensor op for the pt tf.tensor and en tf.tensor
        """

        pt_op, en_op = tf.py_function(func=self.encode,
                                      inp=[pt, en],
                                      Tout=[tf.int64, tf.int64])

        pt_op.set_shape([None])
        en_op.set_shape([None])

        return pt_op, en_op
