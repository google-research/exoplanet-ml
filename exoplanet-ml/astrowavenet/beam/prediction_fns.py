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

"""Beam DoFns for making and saving predictions using an AstroWavenet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os.path

import apache_beam as beam
import numpy as np
import tensorflow as tf

from astrowavenet import astrowavenet_model
from astrowavenet.data import kepler_light_curves
from astrowavenet.util import estimator_util
from tf_util import config_util
from tf_util import example_util


def _get_step_from_checkpoint_path(path):
  """Extracts the global step from a checkpoint path."""
  split_path = path.rsplit("model.ckpt-", 1)
  if len(split_path) != 2:
    raise ValueError("Unrecognized checkpoint path: {}".format(path))
  return int(split_path[1])


class MakePredictionsDoFn(beam.DoFn):
  """Generates predictions for a particular checkpoint."""

  def __init__(self, hparams, dataset_overrides):
    """Initializes the DoFn."""
    self.hparams = hparams
    self.dataset_overrides = dataset_overrides

  def process(self, inputs):
    checkpoint_path, input_file_pattern = inputs
    global_step = _get_step_from_checkpoint_path(checkpoint_path)

    # Create the input_fn.
    dataset_builder = kepler_light_curves.KeplerLightCurves(
        input_file_pattern,
        mode=tf.estimator.ModeKeys.PREDICT,
        config_overrides=self.dataset_overrides)
    tf.logging.info("Dataset config: %s",
                    config_util.to_json(dataset_builder.config))
    input_fn = estimator_util.create_input_fn(dataset_builder)

    # Create the estimator.
    estimator = estimator_util.create_estimator(astrowavenet_model.AstroWaveNet,
                                                self.hparams)

    # Generate predictions.
    for predictions in estimator.predict(
        input_fn, checkpoint_path=checkpoint_path):
      # Add global_step.
      predictions["global_step"] = global_step

      # Squeeze and un-pad the sequences.
      weights = np.squeeze(predictions["seq_weights"])
      real_length = len(weights)
      while real_length > 0 and weights[real_length - 1] == 0:
        real_length -= 1
      for name, value in predictions.items():
        value = np.squeeze(predictions[name])
        if value.shape:
          value = value[0:real_length]
          predictions[name] = value

      yield predictions


class SaveLossesDoFn(beam.DoFn):
  """Writes losses for a particular global step to a csv file."""

  def __init__(self, output_dir):
    self.output_dir = output_dir

  def start_bundle(self):
    if not tf.gfile.Exists(self.output_dir):
      tf.gfile.MakeDirs(self.output_dir)

  def process(self, inputs):
    # Unpack the inputs and sort predictions by loss.
    global_step, all_predictions = inputs
    all_predictions = sorted(all_predictions, key=lambda p: p["mean_loss"])
    if not all_predictions:
      return

    # Write the CSV.
    csv_filename = os.path.join(self.output_dir, "{}.csv".format(global_step))
    with tf.gfile.Open(csv_filename, "w") as f:
      fieldnames = ["example_id", "mean_loss"]
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      for predictions in all_predictions:
        writer.writerow({
            "example_id": predictions["example_id"],
            "mean_loss": predictions["mean_loss"],
        })


class SavePredictionsDoFn(beam.DoFn):
  """Writes predictions for a particular example to a TFRecord file."""

  def __init__(self, output_dir):
    self.output_dir = output_dir

  def start_bundle(self):
    if not tf.gfile.Exists(self.output_dir):
      tf.gfile.MakeDirs(self.output_dir)

  def process(self, inputs):
    # Unpack the inputs and sort predictions by global step.
    example_id, all_predictions = inputs
    all_predictions = sorted(all_predictions, key=lambda p: p["global_step"])
    if not all_predictions:
      return

    filename = os.path.join(self.output_dir, "{}.tfrecord".format(example_id))
    with tf.python_io.TFRecordWriter(filename) as writer:
      for predictions in all_predictions:
        ex = tf.train.Example()
        for name, value in predictions.items():
          if name == "example_id":
            continue
          if not np.shape(value):
            value = [value]
          example_util.set_feature(ex, name, value)
        writer.write(ex.SerializeToString())
