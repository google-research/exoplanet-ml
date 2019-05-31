from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class RvModel(object):
    """A TensorFlow model for predicting radial velocities."""

    def __init__(self, features, hparams, mode):
        """Basic setup.

        The actual TensorFlow model is constructed in build().

        Args:
          features: Dictionary containing "ccf_data" and "label".
          hparams: A ConfigDict of hyperparameters for building the model.
          mode: A tf.estimator.ModeKeys to specify whether the graph should be built
            for training, evaluation or prediction.

        Raises:
          ValueError: If mode is invalid.
        """
        valid_modes = [
            tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL,
            tf.estimator.ModeKeys.PREDICT
        ]
        if mode not in valid_modes:
            raise ValueError("Expected mode in {}. Got: {}".format(valid_modes, mode))

        self.features = features
        self.hparams = hparams
        self.mode = mode

        self.ccf_data = self.features["ccf_data"]
        self.label = self.features["label"]

    def build_network(self):
        """Builds the neural network."""
        net = self.ccf_data

        # Reshape [length] -> [length, 1].
        net = tf.expand_dims(net, -1)

        for i in hparams.conv_block_filters:
            for _ in range(hparams.conv_layers_per_block):
                conv_op = tf.keras.layers.Conv1D(filters=i, kernel_size=hparams.kernel_size, padding='same',
                                                 activation=tf.nn.relu)
                net = conv_op(net)
            max_pool = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
            net = max_pool(net)

        for i in hparams.final_conv_num_filters:
            conv_op = tf.keras.layers.Conv1D(filters=i, kernel_size=hparams.kernel_size, padding='same',
                                             activation=tf.nn.relu)
            net = conv_op(net)
            flatten = tf.keras.layers.Flatten()
            net = flatten(net)

        for i in hparams.dense_num_layers:
            dense = tf.keras.layers.Dense(i, activation=tf.nn.relu)
            net = dense(net)

        # last output layer
        output = tf.keras.layers.Dense(1)
        net = output(net)

        self.predicted_rv = net

    def build_losses(self):
        """Builds the training losses."""
        self.batch_losses = (self.label - self.predicted_rv) ** 2
        self.num_examples = tf.shape(self.label)[0]
        self.total_loss = tf.reduce_sum(self.batch_losses) / tf.cast(self.num_examples, tf.float32)

    def build(self):
        """Creates all ops for training, evaluation or inference."""
        self.global_step = tf.train.get_or_create_global_step()
        self.build_network()
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.build_losses()
