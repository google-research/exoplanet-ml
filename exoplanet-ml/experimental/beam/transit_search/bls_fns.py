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

"""Beam DoFns for running Box Least Squares and processing the output."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import apache_beam as beam
from apache_beam.metrics import Metrics
import numpy as np

from box_least_squares import box_least_squares_pb2 as bls_pb2
from box_least_squares.python import box_least_squares
from experimental.beam.transit_search import bls_scorer
from light_curve import light_curve_pb2


def _max_duration(period, density_star):
  return (period * 365.25**2 / (np.pi**3 * density_star * 215**3))**(1 / 3)


class GeneratePeriodogramDoFn(beam.DoFn):
  """Generates the BLS periodogram for a light curve."""

  def __init__(self, all_periods, all_nbins, weight_min_factor,
               duration_density_min, duration_min_days, duration_density_max,
               duration_min_fraction):
    """Initializes the DoFn."""
    self.all_periods = all_periods
    self.all_nbins = all_nbins
    self.max_nbins = max(self.all_nbins)
    self.weight_min_factor = weight_min_factor
    self.duration_density_min = duration_density_min
    self.duration_min_days = duration_min_days
    self.duration_density_max = duration_density_max
    self.duration_min_fraction = duration_min_fraction

  def process(self, inputs):
    """Generates the BLS periodogram for a light curve.

    Args:
      inputs: A tuple (key, light_curve_pb2.LightCurve)

    Yields:
      A tuple (key, box_least_squares_pb2.Periodogram)
    """
    Metrics.counter(self.__class__.__name__, "inputs-seen").inc()

    # Unpack the light curve.
    lc = inputs["light_curve"]
    time = np.array(lc.light_curve.time, dtype=np.float)
    flux = np.array(lc.light_curve.flux, dtype=np.float)
    norm_curve = np.array(lc.light_curve.norm_curve, dtype=np.float)
    flux /= norm_curve  # Normalize flux.

    # Fit periodogram.
    bls = box_least_squares.BoxLeastSquares(time, flux, capacity=self.max_nbins)
    results = []
    for period, nbins in itertools.izip(self.all_periods, self.all_nbins):
      bin_width = period / nbins

      # Compute the minimum number of bins for a transit.
      duration_min = 0
      if self.duration_density_max:
        duration_min = self.duration_min_fraction * _max_duration(
            period, density_star=self.duration_density_max)
      if self.duration_min_days:
        duration_min = max(self.duration_min_days, duration_min)
      width_min = int(np.maximum(1, np.floor(duration_min / bin_width)))

      # Compute the maximum number of bins for a transit.
      if self.duration_density_min:
        duration_max = _max_duration(
            period, density_star=self.duration_density_min)
        width_max = int(np.ceil(duration_max / bin_width))
      else:
        width_max = int(np.ceil(0.25 * nbins))

      weight_min = self.weight_min_factor * width_min / nbins
      weight_max = 1

      options = bls_pb2.BlsOptions(
          width_min=width_min,
          width_max=width_max,
          weight_min=weight_min,
          weight_max=weight_max)
      try:
        result = bls.fit(period, nbins, options)
      except ValueError:
        Metrics.counter(self.__class__.__name__,
                        "bls-error-{}".format(inputs["kepler_id"])).inc()
        return

      results.append(result)

    inputs["periodogram"] = bls_pb2.Periodogram(results=results)

    yield inputs


def score_method_args_str(name, args):
  args_str = ",".join(["{}={}".format(k, args[k]) for k in sorted(args.keys())])
  return "{}:{}".format(name, args_str) if args_str else name


class TopResultsDoFn(beam.DoFn):
  """Computes the top scoring results of a BLS periodogram."""

  def __init__(self, score_methods, ignore_negative_depth):
    self.score_methods = score_methods
    self.ignore_negative_depth = ignore_negative_depth

  def process(self, inputs):
    # Unpack the inputs.
    results = list(inputs["periodogram"].results)
    scorer = bls_scorer.BlsScorer(
        results, ignore_negative_depth=self.ignore_negative_depth)

    top_results = bls_pb2.TopResults()
    for name, args in self.score_methods:
      score, result = scorer.score(name, **args)

      # Gather name and args into a single string.
      score_method = score_method_args_str(name, args)

      # top_results.scored_results.add(
      #     result=result, score_method=score_method, score=score)
      scored_result = bls_pb2.ScoredResult(
          result=result, score_method=score_method, score=score)
      top_results.scored_results.extend([scored_result])

    inputs["top_results"] = top_results
    yield inputs


class GetTopResultDoFn(beam.DoFn):
  """Computes the top scoring results of a BLS periodogram."""

  def __init__(self, score_method):
    self.top_detection_score_method = score_method_args_str(*score_method)

  def process(self, inputs):
    # TODO(shallue): eventually stop outputting TopResults, and just do a
    # ScoredResult.
    top_result = None
    for scored_result in inputs["top_results"].scored_results:
      if scored_result.score_method == self.top_detection_score_method:
        top_result = scored_result

    if top_result is None:
      raise ValueError(
          "Score method {} not found".format(self.top_detection_score_method))

    inputs["top_result"] = top_result
    yield inputs


class PostProcessForNextDetectionDoFn(beam.DoFn):
  """Post processes for the next detection."""

  def __init__(self, score_threshold=None):
    self.score_threshold = score_threshold

  def process(self, inputs):
    top_result = inputs["top_result"]
    if not self.score_threshold or top_result.score >= self.score_threshold:
      events_to_remove = list(inputs["light_curve"].removed_events)
      events_to_remove.append(
          light_curve_pb2.PeriodicEvent(
              period=top_result.result.period,
              t0=top_result.result.epoch,
              duration=top_result.result.duration))
      outputs = {
          "kepler_id": inputs["kepler_id"],
          "raw_light_curve": inputs["raw_light_curve"],
          "events_to_remove": events_to_remove
      }
      if "light_curve_for_predictions" in inputs:
        outputs["light_curve_for_predictions"] = inputs[
            "light_curve_for_predictions"]

      yield outputs
