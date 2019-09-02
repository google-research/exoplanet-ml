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

"""DoFns for making predictions on BLS detections with an AstroNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os.path

import apache_beam as beam
from apache_beam.metrics import Metrics
import numpy as np
import pandas as pd
import tensorflow as tf

from astronet import models
from astronet.data import preprocess
from tf_util import configdict


class MakePredictionsDoFn(beam.DoFn):
  """Generates predictions from a trained AstroNet model."""

  def __init__(self, model_name, model_dir, config_name=None):
    """Initializes the DoFn.

    Args:
      model_name: Name of the model class.
      model_dir: Directory containing a model checkpoint.
      config_name: Optional name of the model configuration. If not specified,
        the file 'config.json' in model_dir is used.
    """
    # Look up the model class.
    model_class = models.get_model_class(model_name)

    # Find the latest checkpoint.
    checkpoint_file = tf.train.latest_checkpoint(model_dir)
    if not checkpoint_file:
      raise ValueError("No checkpoint file found in: {}".format(model_dir))

    # Get the model configuration.
    if config_name:
      config = models.get_model_config(model_name, config_name)
    else:
      with tf.gfile.Open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    config = configdict.ConfigDict(config)

    self.model_class = model_class
    self.checkpoint_file = checkpoint_file
    self.config = config

  def start_bundle(self):
    # Build the model.
    g = tf.Graph()
    with g.as_default():
      example_placeholder = tf.placeholder(tf.string, shape=[])
      parsed_features = tf.parse_single_example(
          example_placeholder,
          features={
              feature_name: tf.FixedLenFeature([feature.length], tf.float32)
              for feature_name, feature in self.config.inputs.features.items()
          })
      features = {}
      for feature_name, value in parsed_features.items():
        value = tf.expand_dims(value, 0)  # Add batch dimension.
        if self.config.inputs.features[feature_name].is_time_series:
          features.setdefault("time_series_features", {})[feature_name] = value
        else:
          features.setdefault("aux_features", {})[feature_name] = value

      model = self.model_class(
          features=features,
          labels=None,
          hparams=self.config.hparams,
          mode=tf.estimator.ModeKeys.PREDICT)
      model.build()
      saver = tf.train.Saver()

    sess = tf.Session(graph=g)
    saver.restore(sess, self.checkpoint_file)
    tf.logging.info("Successfully loaded checkpoint %s at global step %d.",
                    self.checkpoint_file, sess.run(model.global_step))

    self.example_placeholder = example_placeholder
    self.model = model
    self.session = sess

  def finish_bundle(self):
    self.session.close()

  def process(self, inputs):
    """Generates predictions for a single light curve."""
    lc = inputs["light_curve_for_predictions"]
    time = np.array(lc.light_curve.time, dtype=np.float)
    flux = np.array(lc.light_curve.flux, dtype=np.float)
    norm_curve = np.array(lc.light_curve.norm_curve, dtype=np.float)
    flux /= norm_curve  # Normalize flux.

    # Extract the TCE.
    top_result = inputs["top_result"]
    example = None
    if top_result.HasField("fitted_params"):
      tce = {
          "tce_period": top_result.fitted_params.period,
          "tce_duration": top_result.fitted_params.t0,
          "tce_time0bk": top_result.fitted_params.duration,
      }
      try:
        example = preprocess.generate_example_for_tce(time, flux, tce)
      except ValueError:
        Metrics.counter(self.__class__.__name__,
                        "generate-example-failures").inc()

    if example is None:
      prediction = -1
      serialized_example = tf.train.Example().SerializeToString()
    else:
      serialized_example = example.SerializeToString()
      prediction = self.session.run(
          self.model.predictions,
          feed_dict={self.example_placeholder: serialized_example})[0][0]

    inputs["prediction"] = prediction
    inputs["serialized_example"] = serialized_example

    yield inputs


class ToCsvDoFn(beam.DoFn):
  """Converts predictions to CSV format."""

  def __init__(self, planet_num=-1):
    self.columns = [
        ("kepid", lambda inputs: inputs["kepler_id"]),
        ("planet_num", lambda inputs: inputs.get("planet_num", planet_num)),
        ("prediction", lambda inputs: inputs["prediction"]),
        ("period", lambda inputs: inputs["top_result"].result.period),
        ("duration", lambda inputs: inputs["top_result"].result.duration),
        ("epoch", lambda inputs: inputs["top_result"].result.epoch),
        ("score_method", lambda inputs: inputs["top_result"].score_method),
        ("score", lambda inputs: inputs["top_result"].score),
        ("depth", lambda inputs: inputs["top_result"].result.depth),
        ("baseline", lambda inputs: inputs["top_result"].result.baseline),
        ("complete_transits", lambda inputs: inputs["complete_transits"]),
        ("partial_transits", lambda inputs: inputs["partial_transits"]),
        ("nbins", lambda inputs: inputs["top_result"].result.nbins),
        ("bls_start",
         lambda inputs: inputs["top_result"].result.bls_result.start),
        ("bls_width",
         lambda inputs: inputs["top_result"].result.bls_result.width),
        ("bls_r", lambda inputs: inputs["top_result"].result.bls_result.r),
        ("bls_s", lambda inputs: inputs["top_result"].result.bls_result.s),
        ("bls_power",
         lambda inputs: inputs["top_result"].result.bls_result.power),
        ("width_min",
         lambda inputs: inputs["top_result"].result.options.width_min),
        ("width_max",
         lambda inputs: inputs["top_result"].result.options.width_max),
        ("weight_min",
         lambda inputs: inputs["top_result"].result.options.weight_min),
        ("weight_max",
         lambda inputs: inputs["top_result"].result.options.weight_max),
    ]

  def csv_header(self):
    return ",".join([column[0] for column in self.columns])

  def process(self, inputs):
    df = pd.DataFrame([
        collections.OrderedDict(
            [(name, fn(inputs)) for name, fn in self.columns])
    ])
    yield df.to_csv(header=False, index=False).strip()
