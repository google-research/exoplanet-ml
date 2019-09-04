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

r"""Beam pipeline for combining and reshuffling TFRecord files."""

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

flags.DEFINE_string("input_file_patterns", None,
                    "Comma-separated file patterns of input TFRecords.")

flags.DEFINE_string("output_dir", None, "Base output directory.")

flags.DEFINE_string("output_name", None, "Base name of output TFRecords.")

flags.DEFINE_integer("num_shards", None, "Number of output shards.")

FLAGS = flags.FLAGS


_LABEL_COLUMN = "av_training_set"


class CountLabelsDoFn(beam.DoFn):
  """Counts the labels in tf.Examples."""

  def get_counter(self, name):
    return beam.metrics.Metrics.counter(self.__class__.__name__, name)

  def process(self, example):
    """Counts a single tf.Example and passes it through to the next stage."""
    self.get_counter("examples-total").inc()
    label = example_util.get_bytes_feature(example, _LABEL_COLUMN)[0]
    self.get_counter("examples-{}".format(label)).inc()
    yield example


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  def pipeline(root):
    """Beam pipeline for preprocessing open images."""
    assert FLAGS.input_file_patterns
    assert FLAGS.output_dir
    assert FLAGS.output_name
    assert FLAGS.num_shards

    # Create Pipeline.
    tfrecords = []
    for i, file_pattern in enumerate(FLAGS.input_file_patterns.split(",")):
      logging.info("Reading TFRecords from %s", file_pattern)
      stage_name = "read_tfrecords_{}".format(i)
      tfrecords.append(root | stage_name >> beam.io.tfrecordio.ReadFromTFRecord(
          file_pattern, coder=beam.coders.ProtoCoder(tf.train.Example)))

    # pylint: disable=expression-not-assigned
    (tfrecords
     | "flatten" >> beam.Flatten()
     | "count_labels" >> beam.ParDo(CountLabelsDoFn())
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
