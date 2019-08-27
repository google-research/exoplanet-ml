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

"""Beam DoFns for creating visualizations using an AstroWavenet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path

import apache_beam as beam
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class MakePredictionPlotDoFn(beam.DoFn):
  """Creates plots of AstroWavenet predictions."""

  def __init__(self,
               output_dir,
               max_seq_length=10000,
               width=12,
               height=6,
               first_plot_chunk_only=False):
    self.output_dir = output_dir
    self.max_seq_length = max_seq_length
    self.width = width
    self.height = height
    self.first_plot_chunk_only = first_plot_chunk_only

  def _make_plot(self, observation_no, target, loc, scale, title, filename):
    # Create the figure.
    fig, ax = plt.subplots(figsize=(self.width, self.height))

    # Plot the ground truth.
    ax.plot(observation_no, target, ".", alpha=0.7, label="Ground Truth")
    # Freeze axis limits for consistency between different figures.
    ax.autoscale(False)
    # Plot the means.
    ax.plot(
        observation_no, loc, ".", alpha=0.3, color="green", label="Prediction")
    # Plot one standard deviation.
    ax.fill_between(
        observation_no,
        loc - scale,
        loc + scale,
        alpha=0.2,
        color="green",
        label="Scale")

    # Set axis properties.
    ax.set_title(title)
    ax.set_ylabel("Flux")
    ax.set_xlabel("Observation Number")
    ax.legend()

    with tf.gfile.Open(filename, "w") as f:
      fig.savefig(f)
    plt.close(fig)

  def process(self, predictions):
    # Unpack the predictions
    example_id = predictions["example_id"]
    global_step = predictions["global_step"]
    target = predictions["target"]
    loc = predictions["loc"]
    scale = predictions["scale"]
    observation_no = np.arange(len(target))

    # Create the output directory.
    save_dir = self.output_dir if self.first_plot_chunk_only else os.path.join(
        self.output_dir, str(example_id))
    if not tf.gfile.Exists(save_dir):
      tf.gfile.MakeDirs(save_dir)

    # Plot title.
    title = "{}: Step {}".format(example_id, global_step)

    # Create plots in chunks.
    seq_len = len(target)
    start = 0
    end = min(seq_len, self.max_seq_length)
    while start < len(target):
      filename = os.path.join(
          save_dir,
          str(example_id) + ".png" if self.first_plot_chunk_only else
          "{}-to-{}-{:06d}.png".format(start, end, global_step))

      self._make_plot(observation_no[start:end], target[start:end],
                      loc[start:end], scale[start:end], title, filename)
      start += self.max_seq_length
      end = min(seq_len, end + self.max_seq_length)
      if self.first_plot_chunk_only:
        break


class MakeAnimationDoFn(beam.DoFn):
  """Creates animations of AstroWavenet predictions."""

  def __init__(self,
               output_dir,
               max_seq_length=10000,
               width=12,
               height=6,
               writer="imagemagick",
               image_format="gif"):
    self.output_dir = output_dir
    self.max_seq_length = max_seq_length
    self.width = width
    self.height = height
    self.writer = writer
    self.image_format = image_format

  def start_bundle(self):
    if not tf.gfile.Exists(self.output_dir):
      tf.gfile.MakeDirs(self.output_dir)

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
    artists_tuple = collections.namedtuple("Artists", ["axis", "loc_line"])
    artists = artists_tuple(axis=ax, loc_line=loc_line)

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
    filename = os.path.join(self.output_dir, "{}.{}".format(
        example_id, self.image_format))
    anim.save(filename, writer=self.writer)
