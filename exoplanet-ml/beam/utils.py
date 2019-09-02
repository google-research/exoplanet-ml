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

"""Utilities for Beam pipelines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import apache_beam as beam
from apache_beam.metrics import Metrics
import numpy as np


def write_to_tfrecord(pcollection, key, output_dir, output_name, coder,
                      num_shards):
  """Extracts attributes and writes them to sharded TFRecord files.

  This is a simple wrapper around beam.io.tfrecordio.WriteToTFRecord that first
  extracts the desired attribute from dicts that comprise the input PCollection.

  Args:
    pcollection: A Beam PCollection of dicts or dict-like objects. Each element
      must have an attribute with the specified key.
    key: Name of the attribute to extract from each dict in the input
      PCollection.
    output_dir: The directory to write to.
    output_name: Output file prefix.
    coder: Coder used to encode each record.
    num_shards: The number of files (shards) used for output.

  Returns:
    A WriteToTFRecord transform object.
  """
  extract_stage_name = "extract_{}".format(output_name)
  write_stage_name = "write_{}".format(output_name)
  return (pcollection
          | extract_stage_name >> beam.Map(lambda inputs: inputs[key])
          | write_stage_name >> beam.io.tfrecordio.WriteToTFRecord(
              os.path.join(output_dir, output_name),
              coder=coder,
              num_shards=num_shards))


class TrainValTestPartitionFn(beam.PartitionFn):
  """PartitionFn to separate the output into train, val, and test sets."""

  def __init__(self, key_name, partitions, keys=None, random_seed=123):
    """Initializes the PartitionFn.

    Args:
      key_name: String name of the key; when partition_for(inputs, ...) is
        called, the value of its key is given by inputs[key_name].
      partitions: A dictionary with partition names as keys, and where values
        are either the list of all input keys in that partition (e.g.
        {"train": [0, 1, 2, 3, 4], "val": [5, 6], "test": [7, 8]}), or floating
        values that sum to 1.0 representing the fraction of all keys in each
        partition (e.g. {"train": 0.8, "val": 0.1, "test": 0.1}). In the latter
        case, the 'keys' argument is compulsory.
      keys: List of all possible input keys. Only used if partition is a dict of
        floating values representing the fraction of keys in each partition.
      random_seed: Random seed used for permuting the keys.
    """
    partition_names = sorted(partitions.keys())  # Sort for reproducibility.
    partition_values = [partitions[value] for value in partition_names]

    partition_to_keys = {}
    if np.all([np.issubdtype(type(v), np.floating) for v in partition_values]):
      # Values must sum to 1.0.
      if not np.isclose(np.sum(partition_values), 1.0):
        raise ValueError(
            "Partition values must sum to 1.0. Got {}".format(partitions))
      # Keys must be provided.
      if not keys:
        raise ValueError(
            "Keys are required when providing partition fractions.")
      num_keys = len(keys)
      if len(set(keys)) != num_keys:
        raise ValueError("Keys are not unique.")

      # Randomly shuffle the keys.
      np.random.seed(random_seed)
      keys = np.random.permutation(keys)

      # Create partitions.
      endpoints = np.cumsum(partition_values) * num_keys
      endpoints = np.concatenate([[0], endpoints]).astype(np.int)
      for name, i, j in zip(partition_names, endpoints[:-1], endpoints[1:]):
        partition_to_keys[name] = set(keys[i:j])
      print("Partitioned %d keys into paritions of sizes %s" % (num_keys, {
          name: len(partition_keys)
          for name, partition_keys in partition_to_keys.items()
      }))
    else:
      # Partitions are sets of keys.
      if keys:
        raise ValueError(
            "Keys were provided but partition values were not floats.")

      num_keys = 0
      all_keys = set()
      for partition_name, partition_keys in partitions.items():
        num_keys += len(partition_keys)
        all_keys.update(partition_keys)
        partition_to_keys[partition_name] = set(partition_keys)
      if len(all_keys) != num_keys:
        raise ValueError("Keys are not unique.")

    self.key_name = key_name
    self.partition_to_keys = partition_to_keys
    self.partition_names = partition_names
    self.partition_to_index = {p: i for i, p in enumerate(self.partition_names)}
    self.num_partitions = len(self.partition_names)

  def _get_partition_name(self, inputs):
    """Returns the partition name for a particular input."""
    key = inputs[self.key_name]
    for name in self.partition_to_keys:
      if key in self.partition_to_keys[name]:
        return name
    raise ValueError("Unrecognized key: {}".format(key))

  def partition_for(self, inputs, num_partitions, *args, **kwargs):
    Metrics.counter(self.__class__.__name__, "inputs").inc()
    if num_partitions != self.num_partitions:
      raise ValueError("Expected {} partitions, got: {}".format(
          self.num_partitions, num_partitions))

    partition = self._get_partition_name(inputs)
    Metrics.counter(self.__class__.__name__,
                    "outputs-partition-{}".format(partition)).inc()
    return self.partition_to_index[partition]
