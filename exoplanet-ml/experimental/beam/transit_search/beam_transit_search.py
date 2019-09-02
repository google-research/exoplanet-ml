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

"""Beam pipeline for running transit searches with Box Least Squares."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging as stdlogging
import os.path

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import numpy as np

from tf_util import configdict

from beam import utils
from beam.light_curve import light_curve_fns
from box_least_squares import box_least_squares_pb2 as bls_pb2
from experimental.beam.light_curve import transit_fns
from experimental.beam.transit_search import bls_fns
from experimental.beam.transit_search import kepler_id
# from experimental.beam.transit_search import prediction_fns  # pylint: disable=line-too-long
from light_curve import light_curve_pb2
from tf_util import config_util


flags.DEFINE_string("input_path", None, "Path to file(s) of Kepler ids.")

flags.DEFINE_string("output_dir", None, "Output directory.")

flags.DEFINE_string(
    "config_json", None,
    "JSON string or JSON file containing the pipeline configuration.")

flags.DEFINE_integer("num_output_shards", None, "Number of output shards.")

flags.DEFINE_string("kepler_data_dir", None,
                    "Base folder containing kepler data.")

flags.DEFINE_boolean("save_intermediate_output", False,
                     "Whether to save intermediate outputs (very large).")

# flags.DEFINE_string("astronet_model", None,
#                     "Name of the AstroNet model class.")
#
# flags.DEFINE_string(
#     "astronet_config_name", None,
#     "Name of the AstroNet configuration. Exactly one of "
#     "--astronet_config_name or --astronet_config_json is required.")
#
# flags.DEFINE_string(
#     "astronet_config_json", None,
#     "JSON string or JSON file containing the AstroNet configuration. Exactly "
#     "one of --astronet_config_name or --astronet_config_json is required.")
#
# flags.DEFINE_string("astronet_model_dir", None,
#                     "Directory containing an AstroNet checkpoint.")


FLAGS = flags.FLAGS


# pylint: disable=expression-not-assigned


def _choose_periods_andrew(period_min, period_max, total_time_days,
                           sampling_factor, r_star_r_sun):
  coeff = 365.25**(2/3) / (np.pi * 215)
  a = coeff * r_star_r_sun**(-1/3)/(sampling_factor * total_time_days)
  num_periods = np.ceil(3*(period_min**(-1/3) - period_max**(-1/3))/a)
  return (period_min**(-1/3) - a * np.arange(num_periods) / 3)**-3


def _choose_periods_uniform_freq(period_min, period_max, num):
  # Convert bounds to frequency
  minimum_frequency = 1.0 / period_max
  maximum_frequency = 1.0 / period_min

  df = (maximum_frequency - minimum_frequency) / (num - 1)
  return 1.0 / (maximum_frequency - df * np.arange(num))


def _choose_nbins_andrew(period, density_star, min_num_bins, num_transit_bins):
  # 13.2189 when density_star = 10.
  coeff = ((np.pi**3 * density_star * 215**3) / 365.25**2)**(1 / 3)
  return int(
      max(np.ceil(num_transit_bins * coeff * period**(2 / 3)), min_num_bins))


def _write_output(pcollection, output_name, value_name, value_coder):
  """Convenience function for writing the output."""
  return utils.write_to_tfrecord(
      pcollection,
      output_dir=FLAGS.output_dir,
      output_name=output_name,
      value_name=value_name,
      value_coder=value_coder,
      num_shards=FLAGS.num_output_shards)


def main(unused_argv):
  stdlogging.getLogger().setLevel(stdlogging.INFO)

  def pipeline(root):
    """Beam pipeline for running transit searches with Box Least Squares."""
    # Parse config.
    config = configdict.ConfigDict(config_util.parse_json(FLAGS.config_json))

    # Choose periods.
    period_min = config.period_min
    period_max = config.period_max
    period_sampling_args = config.period_sampling_args or {}
    if config.period_sampling_method == "andrew":
      choose_periods = _choose_periods_andrew
    elif config.period_sampling_method == "uniform_frequency":
      choose_periods = _choose_periods_uniform_freq
    elif config.period_sampling_method == "logarithmic":
      choose_periods = np.geomspace
    elif config.period_sampling_method == "uniform_period":
      choose_periods = np.linspace
    else:
      raise ValueError("Unrecognized period_sampling_method: {}".format(
          config.period_sampling_method))

    all_periods = choose_periods(period_min, period_max, **period_sampling_args)

    # Choose nbins.
    nbins_args = config.nbins_args or {}
    all_nbins = []
    for period in all_periods:
      if config.nbins_method == "andrew":
        all_nbins.append(_choose_nbins_andrew(period, **nbins_args))
      elif config.nbins_method == "constant":
        all_nbins.append(nbins_args["num"])
      else:
        raise ValueError("Unrecognized nbins_method: {}".format(
            config.nbins_method))

    # Write the config.
    config_json = config.to_json(indent=2)
    root | beam.Create([config_json]) | "write_config" >> beam.io.WriteToText(
        os.path.join(FLAGS.output_dir, "config.json"),
        num_shards=1,
        shard_name_template="")

    # Initialize DoFns.
    # TODO(shallue): I think I can pass these as kwargs into ParDo.
    read_light_curve = light_curve_fns.ReadLightCurveDoFn(
        FLAGS.kepler_data_dir,
        injected_group=config.injected_group,
        scramble_type=config.scramble_type,
        invert_light_curves=config.invert_light_curves)

    # process_light_curve_for_astronet = light_curve_fns.ProcessLightCurveDoFn(
    #     gap_width=config.predictions.gap_width,
    #     normalize_method=config.predictions.normalize_method,
    #     normalize_args=config.predictions.normalize_args,
    #     upward_outlier_sigma_cut=config.predictions.upward_outlier_sigma_cut,
    #     output_name="light_curve_for_predictions")

    generate_periodogram = bls_fns.GeneratePeriodogramDoFn(
        all_periods, all_nbins, config.weight_min_factor,
        config.duration_density_min, config.duration_min_days,
        config.duration_density_max, config.duration_min_fraction)

    compute_top_results = bls_fns.TopResultsDoFn(config.score_methods,
                                                 config.ignore_negative_depth)

    get_top_result = bls_fns.GetTopResultDoFn(config.top_detection_score_method)

    fit_transit_params = transit_fns.FitTransitParametersDoFn()

    count_transits = transit_fns.CountTransitsDoFn(
        config.complete_transit_fraction)

    # make_predictions = prediction_fns.MakePredictionsDoFn(
    #     FLAGS.astronet_model, FLAGS.astronet_config_name,
    #     FLAGS.astronet_config_json, FLAGS.astronet_model_dir)

    postprocess_for_next_detection = bls_fns.PostProcessForNextDetectionDoFn(
        score_threshold=config.top_detection_score_threshold)

    # Read Kepler IDs.
    # Output: PCollection({"kepler_id"})
    kep_ids = (
        root
        | "read_kep_ids" >> beam.io.textio.ReadFromText(
            FLAGS.input_path, coder=kepler_id.KeplerIdCoder())
        | "create_input_dicts" >>
        beam.Map(lambda kep_id: {"kepler_id": kep_id.value}))

    # Read light curves.
    # Input: PCollection({"kepler_id"})
    # Output: PCollection({"kepler_id", "raw_light_curve"})
    raw_light_curves = (
        kep_ids
        | "read_light_curves" >> beam.ParDo(read_light_curve))
    # | "process_light_curve_for_astronet" >>
    # beam.ParDo(process_light_curve_for_astronet))

    if FLAGS.save_intermediate_output:
      _write_output(
          raw_light_curves,
          output_name="raw-light-curves",
          value_name="raw_light_curve",
          value_coder=beam.coders.ProtoCoder(light_curve_pb2.RawLightCurve))

    # csv_lines = []
    for planet_num in range(config.max_detections):
      if planet_num > config.clip_downward_outliers_after_planet_num:
        downward_outlier_sigma_cut = config.downward_outlier_sigma_cut
      else:
        downward_outlier_sigma_cut = None

      process_light_curve = light_curve_fns.ProcessLightCurveDoFn(
          gap_width=config.gap_width,
          normalize_method=config.normalize_method,
          normalize_args=config.normalize_args,
          upward_outlier_sigma_cut=config.upward_outlier_sigma_cut,
          downward_outlier_sigma_cut=downward_outlier_sigma_cut,
          remove_events_width_factor=config.remove_events_width_factor)

      # Process light curves.
      # Input: PCollection({
      #   "kepler_id",
      #   "raw_light_curve",
      #   "events_to_remove",  (optional)
      #  })
      # Output: PCollection({
      #   "kepler_id",
      #   "raw_light_curve",
      #   "light_curve",
      # })
      light_curves = (
          raw_light_curves | "process_light_curves-%d" % planet_num >>
          beam.ParDo(process_light_curve))

      # Generate periodograms.
      # Input: PCollection({
      #   "kepler_id",
      #   "raw_light_curve",
      #   "light_curve",
      #  })
      # Output: PCollection({
      #   "kepler_id",
      #   "raw_light_curve",
      #   "light_curve",
      #   "periodogram",
      # })
      periodograms = (
          light_curves | "generate_periodogram-%d" % planet_num >>
          beam.ParDo(generate_periodogram))

      # Compute top results.
      # Input: PCollection({
      #   "kepler_id",
      #   "raw_light_curve",
      #   "light_curve",
      #   "periodogram",
      # })
      # Output: PCollection({
      #   "kepler_id",
      #   "raw_light_curve",
      #   "light_curve",
      #   "periodogram",
      #   "top_results",
      #   "top_result",
      # })
      top_results = (
          periodograms
          | "compute_top_results-%d" % planet_num >>
          beam.ParDo(compute_top_results)
          | "get_top_result-%d" % planet_num >> beam.ParDo(get_top_result)
          | "count_transits-%d" % planet_num >> beam.ParDo(count_transits)
          | "fit_transit_params-%d" % planet_num >>
          beam.ParDo(fit_transit_params))
      # | "make_predictions-%d" % planet_num >> beam.ParDo(make_predictions))

      # csv_lines.append(top_results
      #                 | "extract_csv_%d" % planet_num >> beam.ParDo(
      #                     prediction_fns.ToCsvDoFn(planet_num=planet_num)))

      # Write the outputs.
      _write_output(
          top_results,
          output_name="top-results-%d" % planet_num,
          value_name="top_results",
          value_coder=beam.coders.ProtoCoder(bls_pb2.TopResults))
      # Write the outputs.
      _write_output(
          top_results,
          output_name="scored-result-with-transit-fit-%d" % planet_num,
          value_name="top_result",
          value_coder=beam.coders.ProtoCoder(bls_pb2.ScoredResult))
      if FLAGS.save_intermediate_output:
        _write_output(
            light_curves,
            output_name="light-curves-%d" % planet_num,
            value_name="light_curve",
            value_coder=beam.coders.ProtoCoder(light_curve_pb2.LightCurve))
        _write_output(
            periodograms,
            output_name="periodograms-%d" % planet_num,
            value_name="periodogram",
            value_coder=beam.coders.ProtoCoder(bls_pb2.Periodogram))

      # Process light curves for the next round.
      if planet_num < config.max_detections - 1:
        # Extract detected events.
        # Input: PCollection({
        #   "kepler_id",
        #   "raw_light_curve",
        #   "light_curve",
        #   "periodogram",
        #   "top_results",
        # })
        # Output: PCollection({
        #   "kepler_id",
        #   "raw_light_curve",
        #   "events_to_remove",
        # })
        raw_light_curves = (
            top_results
            | "postprocess-%d" % planet_num >>
            beam.ParDo(postprocess_for_next_detection))

    # (csv_lines
    #  | "flatten_csv_lines" >> beam.Flatten()
    #  | "reshuffle_csv_lines" >> beam.Reshuffle()
    #  | "write_csv" >> beam.io.WriteToText(
    #      os.path.join(FLAGS.output_dir, "predictions.csv"),
    #      num_shards=1,
    #      header=prediction_fns.ToCsvDoFn().csv_header(),
    #      shard_name_template=""))

  pipeline.run()  # result =
  logging.info("Job completed successfully")


if __name__ == "__main__":
  app.run()
