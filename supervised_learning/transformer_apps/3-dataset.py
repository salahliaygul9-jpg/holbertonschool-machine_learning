#!/usr/bin/env python3
''' Dataset class for transformer model '''

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

class Dataset():
    ''' Dataset class '''
    def __init__(self, batch_size, max_len):
        ''' contructor '''
        data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(data_train)
        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)

        def filter_max_length(self, x, y, max_length=max_len):
                ''' filter method '''
                return tf.logical_and(tf.size(x) <= max_length,
                                    tf.size(y) <= max_length)

        self.data_train = self.data_train.filter(filter_max_length)
        self.data_valid = self.data_valid.filter(filter_max_length)

        self.data_train = self.data_train.cache()

        data_size = sum(1 for _ in self.data_train)
        self.data_train = self.data_train.shuffle(data_size)

        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_valid = self.data_valid.padded_batch(batch_size)

        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

    def tokenize_dataset(self, data):
        ''' creates sub-word tokenizers for our dataset '''

        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        ''' encodes a translation into tokens '''
        pt_start_index = self.tokenizer_pt.vocab_size
        pt_end_index = pt_start_index + 1
        en_start_index = self.tokenizer_en.vocab_size
        en_end_index = en_start_index + 1
        pt_tokens = [pt_start_index] + self.tokenizer_pt.encode(
            pt.numpy()) + [pt_end_index]
        en_tokens = [en_start_index] + self.tokenizer_en.encode(
            pt.numpy()) + [en_end_index]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        ''' acts as a tensorflow wrapper for the encode instance method '''
        pt_encoded, en_encoded = tf.py_function(self.encode, [pt, en],
                                                [tf.int64, tf.int64])
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])
        return pt_encoded, en_encoded
