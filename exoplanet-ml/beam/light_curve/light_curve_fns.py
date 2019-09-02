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

from light_curve import kepler_io
from light_curve import light_curve_pb2
from light_curve import util
from third_party.kepler_spline import kepler_spline
from third_party.robust_mean import robust_mean


class ReadLightCurveDoFn(beam.DoFn):
  """Reads the light curve for a particular Kepler ID."""

  def __init__(self,
               kepler_data_dir,
               long_cadence=True,
               quarters=None,
               injected_group=None,
               scramble_type=None,
               invert_light_curves=False):
    self.kepler_data_dir = kepler_data_dir
    self.long_cadence = long_cadence
    self.quarters = quarters
    self.injected_group = injected_group
    self.extension = "INJECTED LIGHTCURVE" if injected_group else "LIGHTCURVE"
    self.scramble_type = scramble_type
    self.invert_light_curves = invert_light_curves

  def process(self, inputs):
    """Reads the light curve of a particular Kepler ID."""
    kep_id = inputs["kepler_id"]

    all_time = None
    all_flux = None
    filenames = kepler_io.kepler_filenames(
        base_dir=self.kepler_data_dir,
        kep_id=kep_id,
        long_cadence=self.long_cadence,
        quarters=self.quarters,
        injected_group=self.injected_group)
    if filenames:
      try:
        all_time, all_flux = kepler_io.read_kepler_light_curve(
            filenames,
            light_curve_extension=self.extension,
            scramble_type=self.scramble_type,
            invert=self.invert_light_curves)
      except (IOError, ValueError) as e:
        raise ValueError("Kepler ID: {}, {}".format(kep_id, e))
    else:
      Metrics.counter(self.__class__.__name__, "no-fits-%s" % kep_id).inc()

    raw_lc = light_curve_pb2.RawLightCurve()
    for time, flux in zip(all_time, all_flux):
      raw_lc.segments.add(time=time, flux=flux)
    inputs["raw_light_curve"] = raw_lc

    yield inputs


class ProcessLightCurveDoFn(beam.DoFn):
  """Normalizes a light curve and removes outliers."""

  def __init__(self,
               gap_width,
               normalize_method,
               normalize_args,
               upward_outlier_sigma_cut=None,
               downward_outlier_sigma_cut=None,
               remove_events_width_factor=1.5,
               output_name="light_curve"):
    """Initializes the DoFn.

    Args:
      gap_width: Minimum gap size (in time units) to split the light curve
        before fitting the normalization curve.
      normalize_method: Method for fitting the normalization curve.
      normalize_args: Arguments passed to the function that computes the
        normalization curve.
      upward_outlier_sigma_cut: Number of standard deviations from the median
        flux value above which upward outliers are removed.
      downward_outlier_sigma_cut: Number of standard deviations from the median
        flux value above which downward outliers are removed.
      remove_events_width_factor: Fraction of the duration to remove when
        removing periodic events.
      output_name: Name of the processed light curve in the output dict.
    """
    self.remove_events_width_factor = remove_events_width_factor
    self.gap_width = gap_width
    self.normalize_method = normalize_method
    self.normalize_args = normalize_args
    self.upward_outlier_sigma_cut = upward_outlier_sigma_cut
    self.downward_outlier_sigma_cut = downward_outlier_sigma_cut
    self.output_name = output_name

  def process(self, inputs):
    """Processes a single light curve."""
    raw_lc = inputs["raw_light_curve"]
    all_time = [np.array(s.time, dtype=np.float64) for s in raw_lc.segments]
    all_flux = [np.array(s.flux, dtype=np.float64) for s in raw_lc.segments]
    length_raw = sum([len(t) for t in all_time])

    # Remove events.
    events_to_remove = inputs.pop("events_to_remove", [])
    if events_to_remove:
      all_time, all_flux = util.remove_events(
          all_time,
          all_flux,
          events_to_remove,
          width_factor=self.remove_events_width_factor,
          include_empty_segments=False)

    if not all_time:
      return  # Removed entire light curve.

    # Split on gaps.
    all_time, all_flux = util.split(
        all_time, all_flux, gap_width=self.gap_width)

    # Mask events.
    events_to_mask = inputs.pop("events_to_mask_for_spline", [])
    if events_to_mask:
      all_masked_time, all_masked_flux = util.remove_events(
          all_time,
          all_flux,
          events=events_to_mask,
          width_factor=self.remove_events_width_factor)
    else:
      all_masked_time = all_time
      all_masked_flux = all_flux

    # Fit normalization curve.
    if self.normalize_method == "spline":
      all_spline, metadata = kepler_spline.fit_kepler_spline(
          all_masked_time, all_masked_flux, **self.normalize_args)
    else:
      raise ValueError("Unrecognized normalize_method: {}".format(
          self.normalize_method))

    # Interpolate the spline between the events removed.
    if events_to_mask:
      all_spline = util.interpolate_masked_spline(all_time, all_masked_time,
                                                  all_spline)

    # Concatenate the results.
    time = np.concatenate(all_time)
    flux = np.concatenate(all_flux)
    norm_curve = np.concatenate(all_spline)

    # Initialize the output.
    light_curve = light_curve_pb2.LightCurve(
        length_raw=length_raw,
        spline_metadata=light_curve_pb2.SplineMetadata(
            bkspace=metadata.bkspace,
            bad_bkspaces=metadata.bad_bkspaces,
            likelihood_term=metadata.likelihood_term,
            penalty_term=metadata.penalty_term,
            bic=metadata.bic,
            masked_events=events_to_mask,
            **self.normalize_args),
        removed_events=events_to_remove)

    # If the normalization curve contains NaNs, we can't normalize those places.
    is_finite = np.isfinite(norm_curve)
    is_not_finite = np.logical_not(is_finite)
    if np.any(is_not_finite):
      light_curve.norm_curve_failures.time[:] = time[is_not_finite]
      light_curve.norm_curve_failures.flux[:] = flux[is_not_finite]
      time = time[is_finite]
      flux = flux[is_finite]
      norm_curve = norm_curve[is_finite]

    # Possibly remove outliers.
    if self.upward_outlier_sigma_cut or self.downward_outlier_sigma_cut:
      norm_flux = flux / norm_curve  # We compute outliers on normalized flux.
      deviation = norm_flux - np.median(norm_flux)

      if self.upward_outlier_sigma_cut:
        is_upward_outlier = np.logical_not(
            robust_mean.robust_mean(
                deviation, cut=self.upward_outlier_sigma_cut)[2])
        np.logical_and(is_upward_outlier, deviation > 0, out=is_upward_outlier)
      else:
        is_upward_outlier = np.zeros_like(deviation, dtype=np.bool)

      if self.downward_outlier_sigma_cut:
        is_downward_outlier = np.logical_not(
            robust_mean.robust_mean(
                deviation, cut=self.downward_outlier_sigma_cut)[2])
        np.logical_and(
            is_downward_outlier, deviation < 0, out=is_downward_outlier)
      else:
        is_downward_outlier = np.zeros_like(deviation, dtype=np.bool)

      is_outlier = np.logical_or(is_upward_outlier, is_downward_outlier)
      is_not_outlier = np.logical_not(is_outlier)
      if np.any(is_outlier):
        light_curve.outliers_removed.time[:] = time[is_outlier]
        light_curve.outliers_removed.flux[:] = flux[is_outlier]
        light_curve.outliers_removed.norm_curve[:] = norm_curve[is_outlier]
        time = time[is_not_outlier]
        flux = flux[is_not_outlier]
        norm_curve = norm_curve[is_not_outlier]

    # Fill the output.
    light_curve.light_curve.time[:] = time
    light_curve.light_curve.flux[:] = flux
    light_curve.light_curve.norm_curve[:] = norm_curve
    inputs[self.output_name] = light_curve

    yield inputs
