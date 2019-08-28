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

r"""Beam pipeline for sampling TFRecord records by Kepler ID."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import tensorflow as tf

from tf_util import example_util

flags.DEFINE_string("input_file_pattern", None,
                    "File pattern of input TFRecords.")

flags.DEFINE_string("output_dir", None, "Base output directory.")

flags.DEFINE_string("output_name", None, "Base name of output TFRecords.")

flags.DEFINE_integer("num_shards", None, "Number of output shards.")

flags.DEFINE_string("kepid_whitelist", None,
                    "Comma-separated list of allowed labels.")

FLAGS = flags.FLAGS


class ProcessExampleDoFn(beam.DoFn):
  """Processes a single tf.Example."""

  def __init__(self, kepid_whitelist):
    self.kepid_whitelist = set(kepid_whitelist)

  def get_counter(self, name):
    return beam.metrics.Metrics.counter(self.__class__.__name__, name)

  def process(self, input_ex):
    """Processes a single tf.Example."""
    self.get_counter("examples-input").inc()
    kepid = example_util.get_int64_feature(input_ex, "kepler_id")[0]
    if kepid in self.kepid_whitelist:
      self.get_counter("examples-output").inc()
      yield input_ex


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  def pipeline(root):
    """Beam pipeline for preprocessing open images."""
    assert FLAGS.input_file_pattern
    assert FLAGS.output_dir
    assert FLAGS.output_name
    assert FLAGS.num_shards
    assert FLAGS.kepid_whitelist

    # Read label whitelist.
    kepid_whitelist = [int(kepid) for kepid in FLAGS.kepid_whitelist.split(",")]
    logging.info("Read Kepid whitelist with %d labels", len(kepid_whitelist))

    # Initialize DoFn.
    process_example = ProcessExampleDoFn(kepid_whitelist)

    # Create Pipeline.
    # pylint: disable=expression-not-assigned
    (root
     | "read_tfrecord" >> beam.io.tfrecordio.ReadFromTFRecord(
         FLAGS.input_file_pattern,
         coder=beam.coders.ProtoCoder(tf.train.Example))
     | "process_examples" >> beam.ParDo(process_example)
     | "reshuffle" >> beam.Reshuffle()
     | "write_tfrecord" >> beam.io.tfrecordio.WriteToTFRecord(
         os.path.join(FLAGS.output_dir, FLAGS.output_name),
         coder=beam.coders.ProtoCoder(tf.train.Example),
         num_shards=FLAGS.num_shards))
    # pylint: enable=expression-not-assigned

  pipeline.run()
  logging.info("Processing complete.")


if __name__ == "__main__":
  app.run(main)
