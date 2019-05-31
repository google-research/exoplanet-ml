from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def create_learning_rate(hparams, global_step):
  """Creates a learning rate Tensor.

  Args:
    hparams: ConfigDict containing the learning rate configuration.
    global_step: The global step Tensor.

  Returns:
    A learning rate Tensor.
  """
  if hparams.get("learning_rate_decay_steps"):
    # Linear decay from hparams.learning_rate to 0.
    learning_rate = tf.train.polynomial_decay(
        learning_rate=hparams.learning_rate,
        global_step=global_step,
        decay_steps=hparams.learning_rate_decay_steps,
        end_learning_rate=0,
        power=1.0)
  else:
    learning_rate = tf.constant(hparams.learning_rate)

  return learning_rate


class ModelFn(object):
  """Class that acts as a callable model function for Estimator train / eval."""

  def __init__(self, model_class, hparams):
    """Initializes the model function.

    Args:
      model_class: Model class.
      hparams: A HParams object containing hyperparameters for building and
        training the model.
    """
    self.model_class = model_class
    self.hparams = hparams

  def __call__(self, features, mode):
    """Builds the model and returns an EstimatorSpec."""
    model = self.model_class(features, self.hparams, mode)
    model.build()

    # Possibly create train_op.
    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      learning_rate = create_learning_rate(self.hparams, model.global_step)
      optimizer = training.create_optimizer(self.hparams, learning_rate)
      train_op = training.create_train_op(model, optimizer)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=model.predicted_rv,
        loss=model.total_loss,
        train_op=train_op)
