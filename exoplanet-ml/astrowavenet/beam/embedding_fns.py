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

"""Beam DoFns for extracting embeddings with an AstroWavenet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import apache_beam as beam
from apache_beam.metrics import Metrics
import numpy as np
import tensorflow as tf

from astrowavenet import astrowavenet_model
from astrowavenet.data import kepler_light_curves
from tf_util import config_util
from tf_util import configdict
from tf_util import example_util


class ExtractEmbeddingsDoFn(beam.DoFn):
  """Generates predictions for a particular checkpoint."""

  def __init__(self,
               model_dir,
               checkpoint_filename=None,
               apply_relu_to_embeddings=False):
    """Initializes the DoFn."""
    config = config_util.parse_json(os.path.join(model_dir, "config.json"))
    config = configdict.ConfigDict(config)

    if checkpoint_filename:
      checkpoint_file = os.path.join(model_dir, checkpoint_filename)
    else:
      checkpoint_file = tf.train.latest_checkpoint(model_dir)
      if not checkpoint_file:
        raise ValueError("No checkpoint file found in: {}".format(model_dir))

    self.config = config
    self.checkpoint_file = checkpoint_file
    self.apply_relu_to_embeddings = apply_relu_to_embeddings

  def start_bundle(self):
    # Build the model.
    g = tf.Graph()
    with g.as_default():
      example_placeholder = tf.placeholder(tf.string, shape=[])
      parsed_features = kepler_light_curves.parse_example(example_placeholder)
      parsed_example_id = parsed_features.pop("example_id")
      parsed_time = parsed_features.pop("time")
      features = {
          # Add extra dimensions: [length] -> [1, length, 1].
          feature_name: tf.reshape(value, [1, -1, 1])
          for feature_name, value in parsed_features.items()
      }
      model = astrowavenet_model.AstroWaveNet(
          features=features,
          hparams=self.config.hparams,
          mode=tf.estimator.ModeKeys.PREDICT)
      model.build()
      saver = tf.train.Saver()

    sess = tf.Session(graph=g)
    saver.restore(sess, self.checkpoint_file)
    tf.logging.info("Successfully loaded checkpoint %s at global step %d.",
                    self.checkpoint_file, sess.run(model.global_step))

    self.example_placeholder = example_placeholder
    self.parsed_example_id = parsed_example_id
    self.parsed_time = parsed_time
    self.model = model
    self.session = sess

  def finish_bundle(self):
    self.session.close()

  def process(self, inputs):
    kepler_id = inputs["kepler_id"]
    example = inputs["wavenet_example"]
    model_output = self.session.run(
        {
            "example_id": self.parsed_example_id,
            "time": self.parsed_time,
            "embedding": self.model.network_output
        },
        feed_dict={self.example_placeholder: example.SerializeToString()})

    # Validate Kepler ID.
    if kepler_id != model_output["example_id"]:
      raise ValueError("Expected Kepler ID {}, got {}".format(
          kepler_id, model_output["example_id"]))

    # Remove embeddings corresponding to missing datapoints - these don't have
    # valid time values.
    time = np.squeeze(model_output["time"])
    embedding = np.squeeze(model_output["embedding"])
    mask = example_util.get_int64_feature(example, "mask").astype(np.bool)
    assert len(time) == len(mask), "len(time)={}, len(mask)={}".format(
        len(time), len(mask))
    assert len(embedding) == len(mask), (
        "len(embedding)={}, len(mask)={}".format(len(time), len(mask)))
    time = time[mask]
    embedding = embedding[mask]
    assert len(time) == np.sum(mask), "len(time)={}, sum(mask)={}".format(
        len(time), np.sum(mask))
    assert np.sum(time <= 0) == 0, "time not positive after applying mask"

    # Possibly apply the ReLu function.
    if self.apply_relu_to_embeddings:
      embedding = np.maximum(np.zeros_like(embedding), embedding)

    inputs.update({"time": time, "embedding": embedding})
    Metrics.counter(self.__class__.__name__, "example-embeddings-output").inc()
    yield inputs
