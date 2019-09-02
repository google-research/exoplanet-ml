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

"""Pipeline for making predictions on BLS detections with an AstroNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging as stdlogging
import os.path

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam

from beam import utils
from beam.light_curve import light_curve_fns
from box_least_squares import box_least_squares_pb2 as bls_pb2
from experimental.beam.transit_search import bls_fns
from experimental.beam.transit_search import prediction_fns

flags.DEFINE_string("input_dir", None, "Output directory.")

flags.DEFINE_integer("detections_per_target", None,
                     "Number of detections made per target.")

flags.DEFINE_string("astronet_model", None, "Name of the AstroNet model class.")

flags.DEFINE_string(
    "astronet_config_name", None,
    "Name of the AstroNet configuration. Exactly one of "
    "--astronet_config_name or --astronet_config_json is required.")

flags.DEFINE_string(
    "astronet_config_json", None,
    "JSON string or JSON file containing the AstroNet configuration. Exactly "
    "one of --astronet_config_name or --astronet_config_json is required.")

flags.DEFINE_string("astronet_model_dir", None,
                    "Directory containing an AstroNet checkpoint.")

flags.DEFINE_string("kepler_data_dir", None,
                    "Base folder containing kepler data.")

flags.DEFINE_string("output_dir", None, "Output directory.")

flags.DEFINE_string("injected_group", None,
                    "Optional. One of 'inj1', 'inj2', 'inj3'.")

flags.DEFINE_float("upward_outlier_sigma_cut", None,
                   "Outlier cut before making predictions.")

flags.DEFINE_float(
    "complete_transit_fraction", 0.5,
    "Fraction of expected in-transit points to count as complete.")

FLAGS = flags.FLAGS

# pylint: disable=expression-not-assigned


def _pair_with_kepid(inputs):
  return (inputs["kepler_id"], inputs)


class PairLightCurveAndDetectionsDoFn(beam.DoFn):
  """Pairs light curves and detections."""

  def process(self, inputs):
    kepid, (light_curve, detections) = inputs
    assert len(light_curve) == 1
    light_curve = light_curve[0]
    assert kepid == light_curve["kepler_id"]
    for detection in detections:
      assert kepid == detection["kepler_id"]
      detection.update(light_curve)
      yield detection


class PrepareInputs(beam.DoFn):

  def __init__(self, planet_num):
    self.planet_num = planet_num

  def process(self, inputs):
    kepler_id, top_results = inputs
    yield kepler_id, {
        "kepler_id": kepler_id,
        "top_results": top_results,
        "planet_num": self.planet_num,
        "tce_id": "%s_%s" % (kepler_id, self.planet_num),
    }


def _write_examples(pcollection):
  """Convenience function for writing serialized TensorFlow examples."""
  return utils.write_to_tfrecord(
      pcollection,
      output_dir=FLAGS.output_dir,
      output_name="examples",
      value_name="serialized_example")


def main(unused_argv):
  stdlogging.getLogger().setLevel(stdlogging.INFO)

  def pipeline(root):
    """Beam pipeline for generating light curve periodograms."""
    # Initialize DoFns.
    read_light_curve = light_curve_fns.ReadLightCurveDoFn(
        FLAGS.kepler_data_dir, injected_group=FLAGS.injected_group)

    get_top_result = bls_fns.GetTopResultDoFn("median_flattened")

    count_transits = light_curve_fns.CountTransitsDoFn(
        FLAGS.complete_transit_fraction)

    process_light_curve = light_curve_fns.ProcessLightCurveDoFn(
        gap_width=0.75,
        normalize_method="spline",
        normalize_args={
            "bkspace_min": 0.5,
            "bkspace_max": 20,
            "bkspace_num": 20,
            "penalty_coeff": 1.0,
        },
        upward_outlier_sigma_cut=FLAGS.upward_outlier_sigma_cut,
        output_name="light_curve_for_predictions")

    make_predictions = prediction_fns.MakePredictionsDoFn(
        FLAGS.astronet_model, FLAGS.astronet_config_name,
        FLAGS.astronet_config_json, FLAGS.astronet_model_dir)

    to_csv = prediction_fns.ToCsvDoFn()

    top_results = []
    for planet_num in range(FLAGS.detections_per_target):
      read_stage_name = "read_top_results-%d" % planet_num
      prepare_inputs_stage_name = "prepare_inputs-%d" % planet_num
      top_results.append(
          root
          | read_stage_name >> beam.io.tfrecordio.ReadFromTFRecord(
              os.path.join(FLAGS.input_dir, "top-results-%d*" % planet_num),
              coder=beam.coders.ProtoCoder(bls_pb2.TopResults))
          | prepare_inputs_stage_name >> beam.ParDo(PrepareInputs(planet_num)))

    # Output: PCollection({
    #    "kepler_id",
    #    "raw_light_curve",
    #    "light_curve_for_predictions",
    # })
    light_curves = (
        # TODO(shallue): replace top_results[0] with getting all keys and
        # deduping and removing the reshuffle.
        top_results[0]
        | "reshuffle_top_results" >> beam.Reshuffle()
        | "get_kepids" >> beam.Map(lambda kv: {"kepler_id": kv[0]})
        | "read_light_curves" >> beam.ParDo(read_light_curve)
        | "process_light_curves" >> beam.ParDo(process_light_curve)
        | "pair_lc_with_kepid" >> beam.Map(_pair_with_kepid))

    all_detections = top_results | "flatten_top_results" >> beam.Flatten()
    detections_and_light_curves = (
        [light_curves, all_detections]
        | "group_by_kepid" >> beam.CoGroupByKey()
        | "pair_light_curves_and_detections" >> beam.ParDo(
            PairLightCurveAndDetectionsDoFn()))

    predictions = (
        detections_and_light_curves
        | "get_top_result" >> beam.ParDo(get_top_result)
        | "count_transits" >> beam.ParDo(count_transits)
        | "make_predictions" >> beam.ParDo(make_predictions))

    # Write predictions
    (predictions | "to_csv" >> beam.ParDo(to_csv)
     | "reshuffle_csv_lines" >> beam.Reshuffle()
     | "write_csv" >> beam.io.WriteToText(
         os.path.join(FLAGS.output_dir, "predictions.csv"),
         num_shards=1,
         header=to_csv.csv_header(),
         shard_name_template=""))

    # Write local and global views.
    _write_examples(predictions)

  pipeline.run()
  logging.info("Job completed successfully")


if __name__ == "__main__":
  app.run()
