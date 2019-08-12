# Copyright 2018 The Exoplanet ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for training an AstroNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf


def _polynomial_decay(initial_value, global_step, decay_steps, end_factor,
                      power):
  """Convenience wrapper around tf.train.polynomial_decay."""
  return tf.train.polynomial_decay(
      learning_rate=initial_value,
      global_step=global_step,
      decay_steps=decay_steps,
      end_learning_rate=end_factor * initial_value,
      power=power)


def create_learning_rate_and_weight_decay(hparams, global_step):
  """Creates the learning rate and weight decay, both with the same schedule.

  Args:
    hparams: ConfigDict containing the learning rate and weight decay
      configurations.
    global_step: The global step Tensor.

  Returns:
    (learning_rate, weight_decay): The learning rate and weight decay, which
    both follow the learning rate decay schedule (if any).
  """
  learning_rate = hparams.learning_rate
  weight_decay = hparams.get("weight_decay", 0.0)

  if hparams.get("learning_rate_decay_steps"):
    decay_schedule = functools.partial(
        _polynomial_decay,
        global_step=global_step,
        decay_steps=hparams.learning_rate_decay_steps,
        end_factor=hparams.learning_rate_end_factor,
        power=hparams.learning_rate_decay_power)
    learning_rate = decay_schedule(learning_rate)
    if weight_decay:
      weight_decay = decay_schedule(weight_decay)

  return learning_rate, weight_decay


def create_optimizer(hparams, global_step, use_tpu=False):
  """Creates a TensorFlow Optimizer.

  Args:
    hparams: ConfigDict containing the optimizer configuration.
    global_step: The global step Tensor.
    use_tpu: If True, the returned optimizer is wrapped in a
      CrossShardOptimizer.

  Returns:
    A TensorFlow optimizer.

  Raises:
    ValueError: If hparams.optimizer is unrecognized.
  """
  optimizer_name = hparams.optimizer.lower()
  optimizer_params = {}
  if optimizer_name == "momentum":
    optimizer_class = tf.train.MomentumOptimizer
    optimizer_params["momentum"] = hparams.get("momentum", 0.9)
    optimizer_params["use_nesterov"] = hparams.get("use_nesterov", False)
  elif optimizer_name == "sgd":
    optimizer_class = tf.train.GradientDescentOptimizer
  elif optimizer_name == "adagrad":
    optimizer_class = tf.train.AdagradOptimizer
  elif optimizer_name == "adam":
    optimizer_class = tf.train.AdamOptimizer
  elif optimizer_name == "rmsprop":
    optimizer_class = tf.RMSPropOptimizer
  else:
    raise ValueError("Unknown optimizer: {}".format(hparams.optimizer))

  # Apply weight decay wrapper.
  optimizer_class = (
      tf.contrib.opt.extend_with_decoupled_weight_decay(optimizer_class))

  # Create optimizer.
  learning_rate, weight_decay = create_learning_rate_and_weight_decay(
      hparams, global_step)
  optimizer = optimizer_class(
      weight_decay=weight_decay,
      learning_rate=learning_rate,
      **optimizer_params)

  if use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  return optimizer


def create_train_op(model, optimizer):
  """Creates a Tensor to train the model.

  Args:
    model: Instance of AstroModel.
    optimizer: Instance of tf.train.Optimizer.

  Returns:
    A Tensor that runs a single training step and returns model.total_loss.
  """
  # Maybe clip gradient norms.
  transform_grads_fn = None
  if model.hparams.get("clip_grad_norm"):
    transform_grads_fn = tf.contrib.training.clip_gradient_norms_fn(
        model.hparams.clip_gradient_norm)

  # Create train op.
  return tf.contrib.training.create_train_op(
      total_loss=model.total_loss,
      optimizer=optimizer,
      global_step=model.global_step,
      transform_grads_fn=transform_grads_fn)
