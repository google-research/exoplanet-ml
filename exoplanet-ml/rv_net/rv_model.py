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
        self.total_loss = None

    def build_network(self):
        """Builds the neural network."""
        net = self.ccf_data

        # Reshape [length] -> [length, 1].
        net = tf.expand_dims(net, -1)

        # create summary object
        summary = []

        for i in self.hparams.conv_block_filters:
            for _ in range(self.hparams.conv_layers_per_block):
                input_shape = net.shape.as_list()
                conv_op = tf.keras.layers.Conv1D(filters=i, kernel_size=self.hparams.kernel_size, padding='same',
                                                 activation=tf.nn.relu)
                net = conv_op(net)
                summary.append("Conv1D-{}-{}. Input shape: {}. Output shape: {}".format(self.hparams.kernel_size, i, input_shape,
                                                                             net.shape.as_list()))
            pool_size = 2
            strides = 2
            max_pool = tf.keras.layers.MaxPool1D(pool_size=pool_size, strides=strides)
            net = max_pool(net)
            summary.append("MaxPool1D-{}. Pool Size: {}. Strides: {}".format(self.hparams.kernel_size, pool_size, strides))

        for i in self.hparams.final_conv_num_filters:
            conv_op = tf.keras.layers.Conv1D(filters=i, kernel_size=self.hparams.kernel_size, padding='same',
                                             activation=tf.nn.relu)
            net = conv_op(net)
            flatten = tf.keras.layers.Flatten()
            net = flatten(net)

        for i in self.hparams.dense_num_layers:
            dense = tf.keras.layers.Dense(i, activation=tf.nn.relu)
            net = dense(net)

        # output layer
        output = tf.keras.layers.Dense(1)
        net = tf.squeeze(output(net))

        self.summary = "\n".join(summary)
        self.predicted_rv = net

    def build_losses(self):
        """Builds the training losses."""
        self.batch_losses = tf.squared_difference(self.predicted_rv, self.label)
        self.total_loss = tf.reduce_mean(self.batch_losses)

    def build(self):
        """Creates all ops for training, evaluation or inference."""
        self.global_step = tf.train.get_or_create_global_step()
        self.build_network()
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.build_losses()

class RvLinearModel(object):
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
        self.total_loss = None

    def build_network(self):
        """Builds linear model."""
        # create summary object

        summary = []

        dense_layer = tf.keras.layers.Dense(1)
        summary.append("Dense-{}-{}. Input shape: {}. Output shape: {}".format(self.hparams.kernel_size, 1, 401,1))

        self.summary = "\n".join(summary)
        self.predicted_rv = dense_layer(self.ccf_data)

    def build_losses(self):
        """Builds the training losses."""
        self.batch_losses = tf.squared_difference(self.predicted_rv, self.label)
        self.total_loss = tf.reduce_mean(self.batch_losses)

    def build(self):
        """Creates all ops for training, evaluation or inference."""
        self.global_step = tf.train.get_or_create_global_step()
        self.build_network()
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.build_losses()
