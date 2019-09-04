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

"""Beam DoFns for manipulating light curves."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
from apache_beam.metrics import Metrics
import numpy as np

from experimental.light_curve import transit_model
from light_curve import periodic_event
from light_curve import util


class FitTransitParametersDoFn(beam.DoFn):
  """Fits transit parameters using a transit model."""

  def process(self, inputs):
    # Unpack the light curve.
    lc = inputs["light_curve"]
    time = np.array(lc.light_curve.time, dtype=np.float)
    flux = np.array(lc.light_curve.flux, dtype=np.float)
    norm_curve = np.array(lc.light_curve.norm_curve, dtype=np.float)
    flux /= norm_curve  # Normalize flux.

    # Unpack the detected TCE.
    top_result = inputs["top_result"]
    period = top_result.result.period
    duration = top_result.result.duration
    t0 = top_result.result.epoch
    planet_radius = np.sqrt(top_result.result.depth)

    # Fit transit model.
    try:
      fitted_params = transit_model.fit_transit_parameters(
          time, flux, t0, period, duration, planet_radius)
    except RuntimeError:
      Metrics.counter(self.__class__.__name__, "transit-model-failed").inc()
      fitted_params = None
    except ValueError:
      # This may indicate an infeasible initial condition. We should debug this.
      Metrics.counter(self.__class__.__name__, "transit-model-ValueError-%s" %
                      inputs["kepler_id"]).inc()
      fitted_params = None

    if fitted_params is not None:
      # Set fitted parameters in the top_result.
      top_result.fitted_params.period = fitted_params["period"]
      top_result.fitted_params.t0 = fitted_params["t0"]
      top_result.fitted_params.duration = fitted_params["duration"]
      top_result.fitted_params.planet_radius = fitted_params["planet_radius"]
      top_result.fitted_params.impact = fitted_params["impact"]
      top_result.fitted_params.u1 = fitted_params["u1"]
      top_result.fitted_params.u2 = fitted_params["u2"]

    yield inputs


class CountTransitsDoFn(beam.DoFn):
  """Counts the number of complete and partial transits in a detection."""

  def __init__(self, complete_transit_fraction):
    self.complete_transit_fraction = complete_transit_fraction

  def compute_transit_stats(self, time, event, cadence, transit_stats):
    try:
      points_per_transit = util.count_transit_points(time, event)
    except ValueError:
      Metrics.counter(self.__class__.__name__,
                      "counting-transit-points-failed").inc()
      points_per_transit = None

    if points_per_transit is not None:
      expected_points_per_transit = event.duration / cadence
      fraction_per_transit = points_per_transit / expected_points_per_transit

      transit_frac = self.complete_transit_fraction
      transit_stats.complete_transit_fraction = transit_frac
      transit_stats.complete_transits = np.sum(
          fraction_per_transit >= transit_frac)
      transit_stats.partial_transits = np.sum(
          fraction_per_transit < transit_frac)

  def process(self, inputs):
    """Counts the number of complete and partial transits in a detection."""
    # Parse the light curves.
    time_raw = np.concatenate([
        np.array(s.time, dtype=np.float64)
        for s in inputs["raw_light_curve"].segments
    ])
    time_processed = np.array(
        inputs["light_curve"].light_curve.time, dtype=np.float64)

    # Extract the TCE.
    # Note that we use the BLS period, duration and t0 rather than fitted
    # params, because we're using this to filter out spurious BLS detections.
    top_result = inputs["top_result"]
    event = periodic_event.Event(
        period=top_result.result.period,
        duration=top_result.result.duration,
        t0=top_result.result.epoch)

    cadence = np.min(np.diff(time_raw))
    top_result.cadence = cadence

    # TODO(shallue): Just use the processed light curve and avoid having a
    # function with an output argument.
    self.compute_transit_stats(time_raw, event, cadence,
                               top_result.transit_stats_raw)
    self.compute_transit_stats(time_processed, event, cadence,
                               top_result.transit_stats_processed)

    yield inputs
