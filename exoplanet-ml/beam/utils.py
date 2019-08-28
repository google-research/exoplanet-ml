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

from absl import logging
import apache_beam as beam
from apache_beam.metrics import Metrics
import numpy as np


class TrainValTestPartitionFn(beam.PartitionFn):
  """PartitionFn to separate the output into train, val, and test sets."""

  def __init__(self, key_name, keys, random_seed=123):
    """Initializes the PartitionFn.

    Args:
      key_name: String name of the key; when partition_for(inputs, ...) is
        called, the value of its key is given by inputs[key_name].
      keys: List of all possible input keys.
      random_seed: Random seed used for permuting the keys.
    """
    # Randomly shuffle the keys.
    np.random.seed(random_seed)
    keys = np.random.permutation(keys)

    # Partition the keys into train (80%), val (10%), test (10%).
    num_keys = len(keys)
    train_cutoff = int(0.80 * num_keys)
    val_cutoff = int(0.90 * num_keys)
    train_keys = set(keys[0:train_cutoff])
    val_keys = set(keys[train_cutoff:val_cutoff])
    test_keys = set(keys[val_cutoff:])
    logging.info(
        "Partitioning %d keys into training (%d), validation (%d) and "
        "test (%d)", num_keys, len(train_keys), len(val_keys), len(test_keys))

    self.key_name = key_name
    self.partition_to_keys = {
        "train": train_keys,
        "val": val_keys,
        "test": test_keys,
    }
    self.partition_names = ["train", "val", "test"]
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
