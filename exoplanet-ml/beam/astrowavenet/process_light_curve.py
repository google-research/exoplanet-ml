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

"""Beam DoFn for processing Kepler light curves into WaveNet inputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
from apache_beam.metrics import Metrics
from astropy.io import fits
import numpy as np
import tensorflow as tf

from light_curve import kepler_io
from light_curve import util
from tf_util import example_util


def _nth_smallest(x, n):
  """Returns the n-th smallest element in the array x."""
  return np.partition(x, n)[n]


class ProcessLightCurveDoFn(beam.DoFn):
  """Reads the light curve for a particular Kepler ID."""

  def __init__(self,
               kepler_data_dir,
               flux_column="PDCSAP_FLUX",
               injected_group=None,
               scramble_type=None,
               invert_light_curves=False,
               upward_outlier_clipping=None,
               downward_outlier_clipping=None,
               clip_lowest_n_values=None,
               normalize_stddev=False):
    """Initializes the DoFn.

    Args:
      kepler_data_dir: Base directory containing Kepler data.
      flux_column: Name of the flux column to extract.
      injected_group: Optional string specifying the injected group. One of
        {'inj1', 'inj2', 'inj3'}.
      scramble_type: Optional string specifying the scramble order. One of
        {'SCR1', 'SCR2', 'SCR3'}.
      invert_light_curves: Whether to reflect light curves around the median
        flux value.
      upward_outlier_clipping: If specified, clip upward flux values to this
        number of multiples of the standard deviation.
      downward_outlier_clipping: If specified, clip downward flux values to this
        number of multiples of the standard deviation.
      clip_lowest_n_values: If specified, clip lowest flux values to the value
        of the nth lowest value.
      normalize_stddev: Whether to divide the flux by the standard deviation.
    """
    self.kepler_data_dir = kepler_data_dir
    self.flux_column = flux_column
    self.injected_group = injected_group
    self.extension = "INJECTED LIGHTCURVE" if injected_group else "LIGHTCURVE"
    self.scramble_type = scramble_type
    self.invert_light_curves = invert_light_curves
    self.upward_outlier_clipping = upward_outlier_clipping
    self.downward_outlier_clipping = downward_outlier_clipping
    self.clip_lowest_n_values = clip_lowest_n_values
    self.normalize_stddev = normalize_stddev

  def _scramble_light_curve(self, all_cadence_no, all_time, all_flux,
                            all_quarters, scramble_type):
    """Scrambles a light curve according to the given scrambling procedure."""
    order = kepler_io.SIMULATED_DATA_SCRAMBLE_ORDERS[scramble_type]
    scr_flux = []
    for quarter in order:
      # Ignore missing quarters in the scramble order.
      if quarter in all_quarters:
        scr_flux.append(all_flux[all_quarters.index(quarter)])

    scr_cadence_no = util.reshard_arrays(all_cadence_no, scr_flux)
    scr_time = util.reshard_arrays(all_time, scr_flux)

    return scr_cadence_no, scr_time, scr_flux

  def _read_kepler_light_curve(self, filenames):
    """Reads cadence numbers, time, and flux for a Kepler target star."""
    # Read light curve data.
    all_cadence_no = []
    all_time = []
    all_flux = []
    all_quarters = []
    for filename in filenames:
      with fits.open(tf.gfile.Open(filename, "rb")) as hdu_list:
        quarter = hdu_list["PRIMARY"].header["QUARTER"]
        light_curve = hdu_list[self.extension].data

      cadence_no = light_curve.CADENCENO
      time = light_curve.TIME
      flux = light_curve[self.flux_column]
      if not cadence_no.size:
        continue  # No data.

      flux_finite = np.isfinite(flux)
      if not np.any(flux_finite):
        continue  # All values are NaN.

      median_flux = np.median(flux[flux_finite])

      # Possibly reflect about the median flux.
      if self.invert_light_curves:
        flux -= 2 * median_flux
        flux *= -1

      # Normalize the flux.
      flux /= median_flux  # Center median at 1.
      flux -= 1  # Center median at 0.

      # Interpolate missing time values. This is only needed if scramble_type is
      # specified (NaN time values typically come with NaN flux values, which
      # are removed anyway, but scrambing decouples NaN time values from NaN
      # flux values).
      if self.scramble_type:
        time = util.interpolate_missing_time(time, cadence_no)

      all_cadence_no.append(cadence_no)
      all_time.append(time)
      all_flux.append(flux)
      all_quarters.append(quarter)

    if self.scramble_type:
      all_cadence_no, all_time, all_flux = self._scramble_light_curve(
          all_cadence_no, all_time, all_flux, all_quarters, self.scramble_type)

    cadence_no = np.concatenate(all_cadence_no)
    time = np.concatenate(all_time)
    flux = np.concatenate(all_flux)

    # Remove timestamps with NaN time or flux values.
    flux_and_time_finite = np.logical_and(np.isfinite(flux), np.isfinite(time))
    cadence_no = cadence_no[flux_and_time_finite]
    time = time[flux_and_time_finite]
    flux = flux[flux_and_time_finite]

    return cadence_no, time, flux

  def process(self, inputs):
    """Reads the light curve of a particular Kepler ID."""
    kep_id = inputs["kepler_id"]
    Metrics.counter(self.__class__.__name__, "inputs").inc()

    # Get light curve filenames.
    filenames = kepler_io.kepler_filenames(
        base_dir=self.kepler_data_dir,
        kep_id=kep_id,
        injected_group=self.injected_group)
    if not filenames:
      Metrics.counter(self.__class__.__name__,
                      "no-fits-{}".format(kep_id)).inc()
      return

    cadence_no, time, flux = self._read_kepler_light_curve(filenames)

    # Additional normalization.
    stddev = 1.4826 * np.median(np.abs(flux))  # 1.4826 * MAD (median is zero).
    if self.upward_outlier_clipping:
      # Clip values greater than n stddev from the median (which is zero).
      flux = np.minimum(flux, self.upward_outlier_clipping * stddev)
    if self.downward_outlier_clipping:
      # pylint: disable=invalid-unary-operand-type
      flux = np.maximum(flux, -self.downward_outlier_clipping * stddev)
    if self.clip_lowest_n_values:
      nth_flux = _nth_smallest(flux, self.clip_lowest_n_values)
      flux = np.maximum(flux, nth_flux)
    if self.normalize_stddev:
      flux /= stddev

    cadence_no, time, flux, mask = util.uniform_cadence_light_curve(
        cadence_no, time, flux)

    ex = tf.train.Example()
    example_util.set_int64_feature(ex, "kepler_id", [kep_id])
    example_util.set_bytes_feature(ex, "flux_column", [self.flux_column])
    example_util.set_bytes_feature(ex, "injected_group", [self.injected_group])
    example_util.set_bytes_feature(ex, "scramble_type", [self.scramble_type])
    example_util.set_float_feature(ex, "time", time)
    example_util.set_float_feature(ex, "flux", flux)
    example_util.set_int64_feature(ex, "cadence_no", cadence_no)
    example_util.set_int64_feature(ex, "mask", mask)
    inputs["example"] = ex

    Metrics.counter(self.__class__.__name__, "outputs").inc()
    yield inputs
