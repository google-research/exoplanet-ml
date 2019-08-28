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

"""Generates animated GIFs using predictions from an AstroWaveNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import json
import logging as stdlogging
import os.path

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from astrowavenet import configurations
from astrowavenet.beam import prediction_fns
from tf_util import configdict

flags.DEFINE_string(
    "input_files", None,
    "Comma-separated list of file patterns matching the TFRecord files in the "
    "evaluation dataset.")

flags.DEFINE_string("output_dir", None, "Base output directory.")

FLAGS = flags.FLAGS


class MakeAnimationDoFn(beam.DoFn):
  """Generates animations from predictions."""

  def __init__(self, output_dir, max_seq_length=10000, width=12, height=6):
    self.output_dir = output_dir
    self.max_seq_length = max_seq_length
    self.width = width
    self.height = height

  def process(self, inputs):
    # Unpack the inputs and sort predictions by global step.
    example_id, all_predictions = inputs
    all_predictions = sorted(all_predictions, key=lambda p: p["global_step"])
    if not all_predictions:
      return

    # Create the figure.
    fig, ax = plt.subplots(figsize=(self.width, self.height))

    # Extract the ground truth from the first prediction and plot it. Note that
    # we must plot ground truth before initializing the line, so the predictions
    # will be overlaid.
    ground_truth_flux = all_predictions[0]["target"][:self.max_seq_length]
    observation_no = np.arange(len(ground_truth_flux))
    ax.plot(
        observation_no, ground_truth_flux, ".", alpha=0.7, label="Ground Truth")

    # Initialize the line of predicted means whose data is updated with every
    # new prediction.
    loc_line = ax.plot([], [],
                       ".",
                       alpha=0.3,
                       color="green",
                       label="Prediction")[0]

    # Freeze axis limits and set labels.
    ax.autoscale(False)
    ax.set_ylabel("Flux")
    ax.set_xlabel("Observation Number")
    ax.legend()

    # Artists are objects that are re-drawn each frame. We make the axis itself
    # an artist because fill_between is tricky: it's a collection on the axis,
    # not an artist. The other artist is the line of predicted means.
    Artists = collections.namedtuple("Artists", ["axis", "loc_line"])  # pylint:disable=invalid-name
    artists = Artists(axis=ax, loc_line=loc_line)

    def update_plot(predictions):
      """Updates the prediction plot for a single global step."""
      # Unpack the predictions.
      loc = predictions["loc"][:self.max_seq_length]
      scale = predictions["scale"][:self.max_seq_length]

      # Update the predicted means.
      loc_line.set_data(observation_no, loc)

      # Update the predicted scales.
      for collection in ax.collections:
        if collection.get_label() == "Scale":
          collection.remove()
      ax.fill_between(
          observation_no,
          loc - scale,
          loc + scale,
          alpha=0.2,
          color="green",
          label="Scale")

      # Update title.
      ax.set_title("{}: Step {}".format(example_id, predictions["global_step"]))

      return artists

    anim = animation.FuncAnimation(
        fig, update_plot, frames=all_predictions, interval=400)
    filename = os.path.join(self.output_dir, "{}.mp4".format(example_id))
    writer = "imagemagick"
    anim.save(filename, writer=writer)


def main(unused_argv):
  stdlogging.getLogger().setLevel(stdlogging.INFO)

  def pipeline(root):
    """Beam pipeline for creating animated GIFs using an AstroWaveNet model."""
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
    input_files = input_files[:1]

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
    make_animations = MakeAnimationDoFn(FLAGS.output_dir)

    # pylint: disable=expression-not-assigned
    (root | beam.Create(itertools.product(checkpoint_paths, input_files))
     | "make_predictions" >> beam.ParDo(make_predictions)
     | "group_by_example_id" >> beam.GroupByKey()
     | "make_animations" >> beam.ParDo(make_animations))
    # pylint: enable=expression-not-assigned

  pipeline.run()
  logging.info("Job completed successfully")


if __name__ == "__main__":
  app.run(main)
