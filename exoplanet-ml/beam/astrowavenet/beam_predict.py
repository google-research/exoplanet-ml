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

"""Generates predictions from an AstroWavenet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import logging as stdlogging
import os.path

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import tensorflow as tf

from astrowavenet import configurations
from astrowavenet.beam import prediction_fns
from astrowavenet.beam import visualize_fns
from tf_util import configdict

flags.DEFINE_string(
    "input_files", None,
    "Comma-separated list of file patterns matching the TFRecord files in the "
    "evaluation dataset.")

flags.DEFINE_string("output_dir", None, "Base output directory.")

flags.DEFINE_string("model_dir", None,
                    "Directory containing model checkpoints.")

flags.DEFINE_string("config_name", "base",
                    "Name of the AstroWaveNet configuration.")

flags.DEFINE_string(
    "config_overrides", "{}",
    "JSON string or JSON file containing overrides to the base configuration.")

flags.DEFINE_boolean("save_losses_per_step", True,
                     "Whether to save a csv of losses for each step.")

flags.DEFINE_boolean("save_plots_per_step", True,
                     "Whether to save plots of the predictions at each step.")

flags.DEFINE_boolean("save_all_predictions", True,
                     "Whether to save model predictions at each step.")

flags.DEFINE_boolean(
    "save_animations", False,
    "Whether to save animations of training for each example.")

FLAGS = flags.FLAGS


def key_by(name):
  """Returns a function that pairs an input dictionary with an item's value."""

  def add_key(x):
    return x[name], x

  return add_key


def main(unused_argv):
  stdlogging.getLogger().setLevel(stdlogging.INFO)

  def pipeline(root):
    """Beam pipeline that generates predictions from an AstroWavenet model."""
    # Read filenames of all checkpoints.
    checkpoint_state = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if not checkpoint_state:
      raise ValueError("Failed to load checkpoint state from {}".format(
          FLAGS.model_dir))
    checkpoint_paths = [
        os.path.join(FLAGS.model_dir, base_name)
        for base_name in checkpoint_state.all_model_checkpoint_paths
    ]
    logging.info("Found %d checkpoints in %s", len(checkpoint_paths),
                 FLAGS.model_dir)

    # Read filenames of all input files.
    input_files = []
    for file_pattern in FLAGS.input_files.split(","):
      matches = tf.gfile.Glob(file_pattern)
      if not matches:
        raise ValueError("Found no files matching {}".format(file_pattern))
      logging.info("Reading from %d files matching %s", len(matches),
                   file_pattern)
      input_files.extend(matches)

    # Parse model configs.
    config = configdict.ConfigDict(configurations.get_config(FLAGS.config_name))
    config_overrides = json.loads(FLAGS.config_overrides)
    for key in config_overrides:
      if key not in ["dataset", "hparams"]:
        raise ValueError("Unrecognized config override: {}".format(key))
    config.hparams.update(config_overrides.get("hparams", {}))

    # Create output directory.
    if not tf.gfile.Exists(FLAGS.output_dir):
      tf.gfile.MakeDirs(FLAGS.output_dir)

    # Initialize DoFns.
    make_predictions = prediction_fns.MakePredictionsDoFn(
        config.hparams, config_overrides.get("dataset"))

    # Create pipeline.
    predictions = (
        root
        | beam.Create(itertools.product(checkpoint_paths, input_files))
        | "make_predictions" >> beam.ParDo(make_predictions))
    predictions_per_example = (
        predictions
        | "key_by_example_id" >> beam.Map(key_by("example_id"))
        | "group_by_example_id" >> beam.GroupByKey())
    predictions_per_step = (
        predictions
        | "key_by_global_step" >> beam.Map(key_by("global_step"))
        | "group_by_global_step" >> beam.GroupByKey())

    # pylint: disable=expression-not-assigned
    if FLAGS.save_losses_per_step:
      save_losses = prediction_fns.SaveLossesDoFn(
          os.path.join(FLAGS.output_dir, "losses"))
      predictions_per_step | "save_losses" >> beam.ParDo(save_losses)
    if FLAGS.save_plots_per_step:
      make_plots = visualize_fns.MakePredictionPlotDoFn(
          os.path.join(FLAGS.output_dir, "prediction_plots"))
      predictions | "make_plots" >> beam.ParDo(make_plots)
    if FLAGS.save_all_predictions:
      save_predictions = prediction_fns.SavePredictionsDoFn(
          os.path.join(FLAGS.output_dir, "predictions"))
      (predictions_per_example
       | "save_predictions" >> beam.ParDo(save_predictions))
    if FLAGS.save_animations:
      make_animations = visualize_fns.MakeAnimationDoFn(
          os.path.join(FLAGS.output_dir, "animations"))
      (predictions_per_example
       | "make_animations" >> beam.ParDo(make_animations))
    # pylint: enable=expression-not-assigned

  pipeline.run()
  logging.info("Job completed successfully")


if __name__ == "__main__":
  app.run(main)
