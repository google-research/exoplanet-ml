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

"""Beam data processing pipeline for AstroNet using AstroWaveNet embeddings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
from absl import logging

import apache_beam as beam
from apache_beam.metrics import Metrics
import numpy as np
import pandas as pd
import tensorflow as tf

from astronet.data import preprocess
from beam import utils
from beam.astrowavenet import embedding_fns
from beam.light_curve import light_curve_fns
from light_curve import binning
from tf_util import configdict
from tf_util import example_util

# pylint: disable=expression-not-assigned

flags.DEFINE_string(
    "config_json", None, "List of JSON files containing the configurations for "
    "each type of TCE (e.g. real, injected, inverted).")

flags.DEFINE_string("model_dir", None,
                    "Directory containing an AstroWaveNet model checkpoint.")

flags.DEFINE_string(
    "checkpoint_filename", None,
    "Optional filename for a specific model checkpoint to use.")

flags.DEFINE_string("output_dir", None,
                    "Directory in which to save the output.")

flags.DEFINE_integer("num_shards_train", 128,
                     "Number of shards for the training sets.")

flags.DEFINE_integer("num_shards_val", 8,
                     "Number of shards for the validation sets.")

flags.DEFINE_integer("num_shards_test", 8,
                     "Number of shards for the test sets.")


FLAGS = flags.FLAGS

_LABEL_COLUMN = "av_training_set"

_GLOBAL_VIEW_NUM_BINS = 2001
_LOCAL_VIEW_NUM_BINS = 201
_LOCAL_VIEW_NUM_DURATIONS = 4


def _parse_configs():
  """Parses the configuration files from FLAGS.config_json."""
  configs = []
  for config_filename in FLAGS.config_json.split(","):
    if tf.gfile.Exists(config_filename):
      with tf.gfile.Open(config_filename) as f:
        config = json.load(f)
    else:
      raise ValueError("Cannot find config file: {}".format(config_filename))
    expected_keys = {
        "name", "kepler_data_dir", "input_event_csv_file", "tce_id_dir",
        "astrowavenet_file_pattern", "injected_group", "scramble_type",
        "invert_light_curves"
    }
    if set(config.keys()) != expected_keys:
      raise ValueError("Expected config keys to be {}, got {}".format(
          expected_keys, config.keys()))

    # Add config settings common to all TCE types.
    config = configdict.ConfigDict(config)
    config.update({
        "gap_width": 0.75,
        "normalize_method": "spline",
        "normalize_args": {
            "bkspace_min": 0.5,
            "bkspace_max": 20,
            "bkspace_num": 20,
            "penalty_coeff": 1.0,
        },
        "model_dir": FLAGS.model_dir,
        "checkpoint_filename": FLAGS.checkpoint_filename,
        "column_value_whitelists": {
            _LABEL_COLUMN: ["PC", "AFP", "NTP", "INV", "INJ1", "INJ2", "SCR1"]
        },
        "emb_views": {
            "global_view_nbins": _GLOBAL_VIEW_NUM_BINS,
            "global_view_bin_width_factor": 1 / _GLOBAL_VIEW_NUM_BINS,
            "local_view_nbins": _LOCAL_VIEW_NUM_BINS,
            "local_view_bin_width_factor":
                (_LOCAL_VIEW_NUM_DURATIONS / _LOCAL_VIEW_NUM_BINS),
            "local_view_num_durations": _LOCAL_VIEW_NUM_DURATIONS,
            "aggr_fn": "sum",
        },
        "flux_views": {
            "global_view_nbins": _GLOBAL_VIEW_NUM_BINS,
            "global_view_bin_width_factor": 1 / _GLOBAL_VIEW_NUM_BINS,
            "local_view_nbins": _LOCAL_VIEW_NUM_BINS,
            "local_view_bin_width_factor": 0.16,
            "local_view_num_durations": _LOCAL_VIEW_NUM_DURATIONS,
            "aggr_fn": "median",
        },
        "apply_relu_to_embeddings": False,
        "align_to_predictions": False,
        "interpolate_missing_time": True,
    })
    configs.append(config)
  return configs


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

  # Add artificial labels for simulated data.
  if config.injected_group:
    assert _LABEL_COLUMN not in events
    events[_LABEL_COLUMN] = config.injected_group.upper()

  if config.invert_light_curves:
    assert _LABEL_COLUMN not in events
    events[_LABEL_COLUMN] = "INV"

  if _LABEL_COLUMN not in events:
    raise ValueError("Expected column '{}' in TCE table, or using a simulated "
                     "dataset.".format(_LABEL_COLUMN))

  # Filter events to whitelisted column values.
  for column, whitelist in config.column_value_whitelists.items():
    allowed_events = events[column].isin(whitelist)
    events = events[allowed_events]

  if events.empty:
    raise ValueError(
        "No events remaining after filtering to whitelists {}".format(
            config.column_value_whitelists))

  logging.info("Filtered to %d events satisfying whitelists %s", len(events),
               config.column_value_whitelists)

  return events


def _key_example_by_kepid(ex):
  kepler_id = example_util.get_int64_feature(ex, "kepler_id")[0]
  return int(kepler_id), ex


def _key_dict_by_kepid(inputs):
  return int(inputs["kepler_id"]), inputs


class _GroupEventsAndExamplesDoFn(beam.DoFn):
  """Processes the output of CoGroupByKey on events and tf.Examples."""

  def inc_counter(self, name):
    Metrics.counter(self.__class__.__name__, name).inc()

  def process(self, inputs):
    kepid, (events, light_curves, examples) = inputs
    self.inc_counter("kepids-seen")
    self.inc_counter("kepids-num-events-{}".format(len(events)))

    if len(light_curves) != 1:
      self.inc_counter("kepids-num-light-curves-{}-skipped".format(
          len(light_curves)))
      return
    if len(examples) != 1:
      self.inc_counter("kepids-num-examples-{}-skipped".format(len(examples)))
      return

    for event in events:
      assert kepid == event["kepid"]
      outputs = {
          "kepler_id": event["kepid"],
          "tce_id": event["tce_id"],
          "event": event,
          "wavenet_example": examples[0],
      }
      outputs.update(light_curves[0])
      yield outputs


def _make_views(tce, time, values, global_view_nbins,
                global_view_bin_width_factor, local_view_nbins,
                local_view_bin_width_factor, local_view_num_durations, aggr_fn):
  """Creates global and local views with embeddings or flux."""
  # Extract event attributes.
  period = tce["tce_period"]
  t0 = tce["tce_time0bk"]
  duration = tce["tce_duration"]
  t_min = -period / 2
  t_max = period / 2

  time, values = preprocess.phase_fold_and_sort_light_curve(
      time, values, period, t0)

  aggr_fn = getattr(np, aggr_fn)

  global_view, global_view_counts = binning.bin_and_aggregate(
      time,
      values,
      num_bins=global_view_nbins,
      bin_width=period * global_view_bin_width_factor,
      x_min=t_min,
      x_max=t_max,
      aggr_fn=aggr_fn)

  local_view, local_view_counts = binning.bin_and_aggregate(
      time,
      values,
      num_bins=local_view_nbins,
      bin_width=duration * local_view_bin_width_factor,
      x_min=max(t_min, -duration * local_view_num_durations),
      x_max=min(t_max, duration * local_view_num_durations),
      aggr_fn=aggr_fn)

  return {
      "global_view": (global_view, global_view_counts),
      "local_view": (local_view, local_view_counts),
  }


class _GenerateExampleDoFn(beam.DoFn):
  """Processes the light curve for a Kepler event and returns a tf.Example."""

  def __init__(self, config):
    self.config = config

  def inc_counter(self, name):
    Metrics.counter(self.__class__.__name__, name).inc()

  def process(self, inputs):
    """Processes the light curve for a Kepler event and returns a tf.Example.

    Args:
      inputs: Dict containing "light_curve" and event".

    Yields:
      A tensorflow.train.Example proto containing event features.
    """
    # Unpack the inputs.
    tce = inputs["event"]

    # Make output proto.
    ex = tf.train.Example()

    # From the FITS files, after processing.
    lc = inputs["light_curve"]
    time = np.array(lc.light_curve.time, dtype=np.float64)
    flux = np.array(lc.light_curve.flux, dtype=np.float64)
    norm_curve = np.array(lc.light_curve.norm_curve, dtype=np.float64)
    flux /= norm_curve

    # Make the flux views.
    flux_views = _make_views(tce, time, flux, **self.config.flux_views)
    for name, (view, counts) in flux_views.items():
      assert view.ndim == 1
      # Empty bins fall back to the global median flux value.
      view = np.where(counts > 0, view, np.median(flux))
      # Center the median at 0 and minimum value at -1.
      view -= np.median(view)
      view /= np.abs(np.min(view))
      # Set features.
      example_util.set_float_feature(ex, "{}_flux".format(name), view)
      example_util.set_float_feature(ex, "{}_flux_counts".format(name), counts)

    # From AstroWaveNet.
    awn_time = inputs["time"]
    embedding = inputs["embedding"]

    # Make the embedding views.
    emb_views = _make_views(tce, awn_time, embedding, **self.config.emb_views)
    for name, (view, counts) in emb_views.items():
      assert view.ndim == 2
      view = np.transpose(view)  # Turn into rows of features.
      for i in range(len(view)):
        example_util.set_float_feature(ex, "{}_emb_{}".format(name, i), view[i])
      example_util.set_float_feature(ex, "{}_emb_counts".format(name), counts)

    # Set other features in `tce`.
    for name, value in tce.items():
      example_util.set_feature(ex, name, [value])

    inputs["example"] = ex
    self.inc_counter("examples-output-{}".format(tce[_LABEL_COLUMN]))
    self.inc_counter("examples-output-total")
    yield inputs


class _CountLabelsDoFn(beam.DoFn):
  """Counts the labels in processed TCEs."""

  def __init__(self, prefix):
    self.prefix = prefix

  def get_counter(self, name):
    return beam.metrics.Metrics.counter(self.__class__.__name__, name)

  def process(self, inputs):
    """Counts a single tf.Example and passes it through to the next stage."""
    self.get_counter("{}-examples-total".format(self.prefix)).inc()
    label = example_util.get_bytes_feature(inputs["example"], _LABEL_COLUMN)[0]
    self.get_counter("{}-examples-{}".format(self.prefix, label)).inc()
    yield inputs


def _write_subset(dataset_name, name, values):
  """Writes the tf.Examples in a subset to TFRecord files."""
  if name == "train":
    num_shards = FLAGS.num_shards_train
  elif name == "val":
    num_shards = FLAGS.num_shards_val
  elif name == "test":
    num_shards = FLAGS.num_shards_test
  else:
    raise ValueError("Unrecognized subset name: {}".format(name))

  # Write the tf.Examples in TFRecord format.
  utils.write_to_tfrecord(
      values,
      output_dir=os.path.join(FLAGS.output_dir, dataset_name),
      output_name=name,
      value_name="example",
      value_coder=beam.coders.ProtoCoder(tf.train.Example),
      num_shards=num_shards,
      stage_name_suffix=dataset_name)


def _process_tces(root, config):
  """Creates a beam pipeline to process a set of TCEs."""
  name = config.name

  # Read input events table.
  events = _read_events(config)

  # Initialize DoFns.
  read_light_curve = light_curve_fns.ReadLightCurveDoFn(
      config.kepler_data_dir,
      injected_group=config.injected_group,
      scramble_type=config.scramble_type,
      invert_light_curves=config.invert_light_curves)
  process_light_curve = light_curve_fns.ProcessLightCurveDoFn(
      gap_width=config.gap_width,
      normalize_method=config.normalize_method,
      normalize_args=config.normalize_args)
  extract_embeddings = embedding_fns.ExtractEmbeddingsDoFn(
      model_dir=config.model_dir,
      checkpoint_filename=config.checkpoint_filename,
      apply_relu_to_embeddings=config.apply_relu_to_embeddings,
      align_to_predictions=config.align_to_predictions,
      interpolate_missing_time=config.interpolate_missing_time)
  generate_example = _GenerateExampleDoFn(config)

  # Read TCE ids corresponding to each partition.
  partition_to_ids = {}
  for subset in ["train", "val", "test"]:
    tce_id_filename = os.path.join(config.tce_id_dir,
                                   "{}_tce_ids.txt".format(subset))
    with tf.gfile.Open(tce_id_filename) as f:
      tce_ids = set([line.strip() for line in f])
    partition_to_ids[subset] = tce_ids
    logging.info("%s: Partition '%s' with %s TCE ids.", name, subset,
                 len(tce_ids))
  partition_fn = utils.TrainValTestPartitionFn("tce_id", partition_to_ids)

  # Create pipeline.
  events_by_kepid = (
      root | "{}-create_event_pcollection".format(name) >> beam.Create(
          [(int(event["kepid"]), event) for _, event in events.iterrows()]))
  kepler_ids = [{"kepler_id": int(kepid)} for kepid in set(events["kepid"])]
  logging.info("%s: Reading light curves for %d unique Kepler IDs", name,
               len(kepler_ids))
  light_curves = (
      root
      | "{}-create_kepid_pcollection".format(name) >> beam.Create(kepler_ids)
      | "{}-read_light_curves".format(name) >> beam.ParDo(read_light_curve)
      |
      "{}-process_light_curves".format(name) >> beam.ParDo(process_light_curve)
      | "{}-key_light_curves_by_kepid".format(name) >>
      beam.Map(_key_dict_by_kepid))
  wavenet_inputs = (
      root
      | "{}-read_wavenet_inputs".format(name) >>
      beam.io.tfrecordio.ReadFromTFRecord(
          config.astrowavenet_file_pattern,
          coder=beam.coders.ProtoCoder(tf.train.Example))
      | "{}-key_examples_by_kepid".format(name) >>
      beam.Map(_key_example_by_kepid))
  results = (
      [events_by_kepid, light_curves, wavenet_inputs]
      | "{}-join_events_and_inputs".format(name) >> beam.CoGroupByKey()
      | "{}-group_events_and_inputs".format(name) >> beam.ParDo(
          _GroupEventsAndExamplesDoFn())
      | "{}-extract_embeddings".format(name) >> beam.ParDo(extract_embeddings)
      | "{}-generate_examples".format(name) >> beam.ParDo(generate_example)
      | "{}-reshuffle".format(name) >> beam.Reshuffle()
      | "{}-partition_results".format(name) >> beam.Partition(
          partition_fn, partition_fn.num_partitions))

  return zip(partition_fn.partition_names, results)


def main(argv):
  del argv  # Unused.
  logging.set_verbosity(logging.INFO)

  def pipeline(root):
    """Beam pipeline for preprocessing Kepler events."""
    # Separately process and write each TCE dataset, and gather all the results.
    configs = _parse_configs()
    subsets = {
        "train": [],
        "val": [],
        "test": [],
    }
    for config in configs:
      output_dir = os.path.join(FLAGS.output_dir, config.name)
      # Write the config.
      config_json = json.dumps(config, indent=2)
      logging.info(config_json)
      (root
       | "{}-create-config".format(config.name) >> beam.Create([config_json])
       | "{}-write_config".format(config.name) >> beam.io.WriteToText(
           os.path.join(output_dir, "config.json"),
           num_shards=1,
           shard_name_template=""))
      # Process TCEs and write each subset.
      results = _process_tces(root, config)
      for subset_name, subset_values in results:
        _write_subset(config.name, subset_name, subset_values)
        subsets[subset_name].append(subset_values)

    # Create one dataset comprising all TCE datasets.
    for subset_name, subset_values in subsets.items():
      combined_subset_values = (
          subset_values
          | "combined-{}-flatten".format(subset_name) >> beam.Flatten()
          | "combined-{}-count_labels".format(subset_name) >> beam.ParDo(
              _CountLabelsDoFn(prefix="combined-{}".format(subset_name)))
          | "combined-{}-reshuffle".format(subset_name) >> beam.Reshuffle())
      _write_subset("combined", subset_name, combined_subset_values)

  pipeline.run()
  logging.info("Preprocessing complete.")


if __name__ == "__main__":
  app.run(main)
