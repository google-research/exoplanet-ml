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

"""Beam data processing pipeline for AstroNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
from absl import logging

import apache_beam as beam
import numpy as np
import pandas as pd
import tensorflow as tf

from astronet.data import preprocess
from beam import utils
from beam.light_curve import light_curve_fns
from light_curve import light_curve_pb2
from tf_util import configdict

# pylint: disable=expression-not-assigned

flags.DEFINE_string(
    "input_event_csv_file", None,
    "CSV file containing table of Kepler events to preprocess.")

flags.DEFINE_string("kepler_data_dir", None,
                    "Base folder containing Kepler data.")

flags.DEFINE_string("injected_group", None,
                    "Optional. One of 'inj1', 'inj2', 'inj3'.")

flags.DEFINE_string("output_dir", None,
                    "Directory in which to save the output.")

flags.DEFINE_integer("num_shards_train", 8,
                     "Number of shards for the training set.")

flags.DEFINE_integer("num_shards_val", 1,
                     "Number of shards for the validation set.")

flags.DEFINE_integer("num_shards_test", 1, "Number of shards for the test set.")

flags.DEFINE_boolean("invert_light_curves", False,
                     "Whether to generate inverted light curves.")

flags.DEFINE_string(
    "light_curve_scramble_type", None,
    "What scrambling procedure to use. One of 'SCR1', 'SCR2', 'SCR3', or None.")

FLAGS = flags.FLAGS

_LABEL_COLUMN = "av_training_set"


def _read_events(config):
  """Reads, filters, and partitions a table of Kepler KOIs.

  Args:
    config: ConfigDict containing the configuration.

  Returns:
    events: A pd.DataFrame containing events as rows.

  Raises:
    ValueError: If TCE ids are not unique.
  """
  with tf.gfile.Open(config.input_event_csv_file) as f:
    events = pd.read_csv(f, comment="#")
  events.tce_duration /= 24  # Convert hours to days.
  logging.info("Read event table with %d rows.", len(events))

  # Add TCE ids.
  events["tce_id"] = events.apply(
      lambda event: "%d_%d" % (event.kepid, event.tce_plnt_num), axis=1)
  if len(set(events["tce_id"])) != len(events):
    raise ValueError("TCE ids are not unique.")

  # Filter events to whitelisted column values.
  for column, whitelist in config.column_value_whitelists.items():
    allowed_events = events[column].isin(whitelist)
    events = events[allowed_events]

  logging.info("Filtered to %d events satisfying whitelists %s", len(events),
               config.column_value_whitelists)

  return events


def _prepare_pipeline_inputs(events, config):
  """Converts a table of Kepler events to a list of pipeline input dicts."""

  def _prepare_event(event):
    """Maps an event to a dict of pipeline inputs."""
    kepler_id = event["kepid"]
    tce_id = "%d_%d" % (kepler_id, event["tce_plnt_num"])
    result = {"kepler_id": kepler_id, "tce_id": tce_id, "event": event}
    if config.remove_event_for_spline:
      result["events_to_mask_for_spline"] = [
          light_curve_pb2.PeriodicEvent(
              period=event.tce_period,
              t0=event.tce_time0bk,
              duration=event.tce_duration)
      ]
    return result

  return [_prepare_event(event) for _, event in events.iterrows()]


class GenerateExampleDoFn(beam.DoFn):
  """Processes the light curve for a Kepler event and returns a tf.Example."""

  def process(self, inputs):
    """Processes the light curve for a Kepler event and returns a tf.Example.

    Args:
      inputs: Dict containing "light_curve" and event".

    Yields:
      A tensorflow.train.Example proto containing event features.
    """
    # Unpack the inputs.
    event = inputs["event"]
    lc = inputs["light_curve"]
    time = np.array(lc.light_curve.time, dtype=np.float64)
    flux = np.array(lc.light_curve.flux, dtype=np.float64)
    norm_curve = np.array(lc.light_curve.norm_curve, dtype=np.float64)

    # Normalize the flux.
    flux /= norm_curve

    # Generate example.
    inputs["example"] = preprocess.generate_example_for_tce(time, flux, event)

    yield inputs


def _key_by_tce_id(inputs):
  tce_id = "%d_%d" % (inputs["kepler_id"], inputs["event"]["tce_plnt_num"])
  return (tce_id, inputs)


def main(argv):
  del argv  # Unused.
  logging.set_verbosity(logging.INFO)

  config = configdict.ConfigDict({
      "input_event_csv_file": FLAGS.input_event_csv_file,
      "kepler_data_dir": FLAGS.kepler_data_dir,
      "injected_group": FLAGS.injected_group,
      "invert_light_curves": FLAGS.invert_light_curves,
      "scramble_type": FLAGS.light_curve_scramble_type,
      "gap_width": 0.75,
      "normalize_method": "spline",
      "normalize_args": {
          "bkspace_min": 0.5,
          "bkspace_max": 20,
          "bkspace_num": 20,
          "penalty_coeff": 1.0,
      },
      "remove_event_for_spline": False,
      "remove_events_width_factor": 1.5,
      "upward_outlier_sigma_cut": None,
      "column_value_whitelists": {
          _LABEL_COLUMN: ["PC", "AFP", "NTP", "INV", "INJ1", "SCR1"]
      },
  })

  def pipeline(root):
    """Beam pipeline for preprocessing Kepler events."""
    # Write the config.
    config_json = json.dumps(config, indent=2)
    root | beam.Create([config_json]) | "write_config" >> beam.io.WriteToText(
        os.path.join(FLAGS.output_dir, "config.json"),
        num_shards=1,
        shard_name_template="")

    # Read input events table.
    events = _read_events(config)

    # Initialize DoFns.
    read_light_curve = light_curve_fns.ReadLightCurveDoFn(
        config.kepler_data_dir,
        injected_group=config.injected_group,
        scramble_type=config.scramble_type,
        invert=config.invert_light_curves)
    process_light_curve = light_curve_fns.ProcessLightCurveDoFn(
        gap_width=config.gap_width,
        normalize_method=config.normalize_method,
        normalize_args=config.normalize_args,
        upward_outlier_sigma_cut=config.upward_outlier_sigma_cut,
        remove_events_width_factor=config.remove_events_width_factor)
    generate_example = GenerateExampleDoFn()
    partition_fn = utils.TrainValTestPartitionFn(
        key_name="tce_id",
        partitions={
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
        },
        keys=events.tce_id.values)

    # Create pipeline.
    pipeline_inputs = _prepare_pipeline_inputs(events, config)
    results = (
        root
        | "create_pcollection" >> beam.Create(pipeline_inputs)
        | "read_light_curves" >> beam.ParDo(read_light_curve)
        | "process_light_curves" >> beam.ParDo(process_light_curve)
        | "generate_examples" >> beam.ParDo(generate_example)
        | "reshuffle" >> beam.Reshuffle()
        | "partition_results" >> beam.Partition(partition_fn,
                                                partition_fn.num_partitions))

    for name, subset in zip(partition_fn.partition_names, results):
      if name == "train":
        num_shards = FLAGS.num_shards_train
      elif name == "val":
        num_shards = FLAGS.num_shards_val
      elif name == "test":
        num_shards = FLAGS.num_shards_test
      else:
        raise ValueError("Unrecognized subset name: %s" % name)

      # Write the tf.Examples in TFRecord format.
      utils.write_to_tfrecord(
          subset,
          output_dir=FLAGS.output_dir,
          output_name=name,
          value_name="example",
          value_coder=beam.coders.ProtoCoder(tf.train.Example),
          num_shards=num_shards)

  pipeline.run()
  logging.info("Preprocessing complete.")


if __name__ == "__main__":
  app.run(main)
