from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class DatasetBuilder(object):
    """Dataset builder class."""

    def __init__(self, file_pattern, hparams, mode, repeat=1):
        """Initializes the dataset builder.

        Args:
          file_pattern: File pattern matching input file shards, e.g.
            "/tmp/train-?????-of-00100".
          hparams: A ConfigDict.
          mode: A tf.estimator.ModeKeys.
          repeat: The number of times to repeat the dataset. If None, the dataset
            will repeat indefinitely.
        """
        valid_modes = [
            tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL,
            tf.estimator.ModeKeys.PREDICT
        ]
        if mode not in valid_modes:
            raise ValueError("Expected mode in {}. Got: {}".format(valid_modes, mode))

        self.file_pattern = file_pattern
        self.hparams = hparams
        self.mode = mode
        self.repeat = repeat

    def __call__(self):
        is_training = self.mode == tf.estimator.ModeKeys.TRAIN

        # Dataset of file names.
        filename_dataset = tf.data.Dataset.list_files(self.file_pattern,
                                                      shuffle=is_training)

        # Dataset of serialized tf.Examples.
        dataset = filename_dataset.flat_map(tf.data.TFRecordDataset)

        # Shuffle in training mode.
        if is_training:
            dataset = dataset.shuffle(self.hparams.shuffle_values_buffer)

        # Possibly repeat.
        if self.repeat != 1:
            dataset = dataset.repeat(self.repeat)

        def _example_parser(serialized_example):
            """Parses a single tf.Example into feature and label tensors."""
            data_fields = {
                self.hparams.ccf_feature_name: tf.FixedLenFeature([161], tf.float32),
                self.hparams.label_feature_name: tf.FixedLenFeature([], tf.float32),
            }
            parsed_fields = tf.parse_single_example(serialized_example, features=data_fields)
            ccf_data = parsed_fields[self.hparams.ccf_feature_name]
            label = parsed_fields[self.hparams.label_feature_name]
            label *= self.hparams.label_rescale_factor  # Rescale the label.
            return {
                "ccf_data": ccf_data,
                "label": label,
            }

        # Map the parser over the dataset.
        dataset = dataset.map(_example_parser, num_parallel_calls=4)

        # Batch results by up to batch_size.
        dataset = dataset.batch(self.hparams.batch_size)

        # Prefetch a few batches.
        dataset = dataset.prefetch(10)

        return dataset
