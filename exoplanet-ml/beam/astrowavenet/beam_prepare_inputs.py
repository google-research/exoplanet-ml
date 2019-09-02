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

"""Beam pipeline for processing Kepler light curves into AstroWaveNet inputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
from absl import logging

import apache_beam as beam
import tensorflow as tf

from astrowavenet.beam import process_light_curve
from beam import utils
from tf_util import configdict

# pylint: disable=expression-not-assigned

flags.DEFINE_string("input_kepid_file", None,
                    "File containing Kepler ids to preprocess.")

flags.DEFINE_string("kepler_data_dir", None,
                    "Base folder containing Kepler data.")

flags.DEFINE_string("injected_group", None,
                    "Optional. One of 'inj1', 'inj2', 'inj3'.")

flags.DEFINE_string(
    "scramble_type", None,
    "What scrambling procedure to use. One of 'SCR1', 'SCR2', 'SCR3', or None.")

flags.DEFINE_boolean("invert_light_curves", False,
                     "Whether to generate inverted light curves.")

flags.DEFINE_string("flux_column", "PDCSAP_FLUX", "Which flux column to read.")

flags.DEFINE_string("output_dir", None,
                    "Directory in which to save the output.")

flags.DEFINE_integer("num_shards_train", 8,
                     "Number of shards for the training set.")

flags.DEFINE_integer("num_shards_val", 1,
                     "Number of shards for the validation set.")

flags.DEFINE_integer("num_shards_test", 1, "Number of shards for the test set.")

flags.DEFINE_integer("upward_outlier_clipping", 5,
                     "Maximum allowed standard deviations above the median.")

flags.DEFINE_integer("downward_outlier_clipping", None,
                     "Maximum allowed standard deviations below the median.")

flags.DEFINE_integer("clip_lowest_n_values", 20,
                     "Number of flux values to clip from the bottom.")

flags.DEFINE_boolean("normalize_stddev", True,
                     "Whether or not to normalize the standard deviation.")

FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.
  logging.set_verbosity(logging.INFO)

  config = configdict.ConfigDict({
      "input_kepid_file": FLAGS.input_kepid_file,
      "kepler_data_dir": FLAGS.kepler_data_dir,
      "flux_column": FLAGS.flux_column,
      "injected_group": FLAGS.injected_group,
      "scramble_type": FLAGS.scramble_type,
      "invert_light_curves": FLAGS.invert_light_curves,
      "upward_outlier_clipping": FLAGS.upward_outlier_clipping,
      "downward_outlier_clipping": FLAGS.downward_outlier_clipping,
      "clip_lowest_n_values": FLAGS.clip_lowest_n_values,
      "normalize_stddev": FLAGS.normalize_stddev,
  })

  def pipeline(root):
    """Beam pipeline for preprocessing Kepler events."""
    if not FLAGS.input_kepid_file:
      raise ValueError("--input_kepid_file is required")
    if not FLAGS.kepler_data_dir:
      raise ValueError("--kepler_data_dir is required")
    if not FLAGS.output_dir:
      raise ValueError("--output_dir is required")

    # Write the config.
    config_json = json.dumps(config, indent=2)
    root | beam.Create([config_json]) | "write_config" >> beam.io.WriteToText(
        os.path.join(FLAGS.output_dir, "config.json"),
        num_shards=1,
        shard_name_template="")

    # Read input Kepler ids.
    with tf.gfile.Open(config.input_kepid_file) as f:
      kep_ids = [int(line.strip()) for line in f]
    logging.info("Read %d Kepler ids from %s", len(kep_ids),
                 config.input_kepid_file)

    # Initialize DoFns.
    process_fn = process_light_curve.ProcessLightCurveDoFn(
        config.kepler_data_dir,
        flux_column=config.flux_column,
        injected_group=config.injected_group,
        scramble_type=config.scramble_type,
        invert_light_curves=config.invert_light_curves,
        upward_outlier_clipping=config.upward_outlier_clipping,
        downward_outlier_clipping=config.downward_outlier_clipping,
        clip_lowest_n_values=config.clip_lowest_n_values,
        normalize_stddev=config.normalize_stddev)
    partition_fn = utils.TrainValTestPartitionFn(
        key_name="kepler_id",
        partitions={
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
        },
        keys=kep_ids)

    # Create pipeline.
    inputs = [{"kepler_id": kep_id} for kep_id in kep_ids]
    results = (
        root
        | "create_pcollection" >> beam.Create(inputs)
        | "process_light_curves" >> beam.ParDo(process_fn)
        | "reshuffle" >> beam.Reshuffle()
        | "partition_results" >> beam.Partition(partition_fn,
                                                partition_fn.num_partitions))

    # Write the outputs in TFRecord format.
    for name, subset in zip(partition_fn.partition_names, results):
      if name == "train":
        num_shards = FLAGS.num_shards_train
      elif name == "val":
        num_shards = FLAGS.num_shards_val
      elif name == "test":
        num_shards = FLAGS.num_shards_test
      else:
        raise ValueError("Unrecognized subset name: {}".format(name))

      utils.write_to_tfrecord(
          subset,
          key="example",
          output_dir=FLAGS.output_dir,
          output_name=name,
          coder=beam.coders.ProtoCoder(tf.train.Example),
          num_shards=num_shards)

  pipeline.run()
  logging.info("Preprocessing complete.")


if __name__ == "__main__":
  app.run(main)
