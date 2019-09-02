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

"""Score functions for selecting the top result in a BLS periodogram."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal


def _midpoint(x):
  return (x[0] + x[-1]) / 2


def _get_aggr_fn(name):
  if name == "mean":
    return np.mean
  elif name == "median":
    return np.median
  elif name == "midpoint":
    return _midpoint
  else:
    raise ValueError("Unrecognized aggr_fn name: %s" % name)


def _linear_bin_endpoints(x, nbins):
  """Computes endpoint indices for evenly spaced bins."""
  # Assume x is sorted in ascensing or descending order.
  first = x[0]
  last = x[-1]
  if first < last:
    comparator = np.greater
  else:
    comparator = np.less

  x_endpoints = np.linspace(first, last, num=nbins + 1)
  index_endpoints = np.zeros(nbins + 1, dtype=np.int)

  # Loop through x. Current bin is x_endpoints[i-1] (inclusive)
  # to x_endpoints[i] (exclusive).
  i = 1
  for j in range(len(x)):
    if comparator(x[j], x_endpoints[i]):
      index_endpoints[i] = j
      i += 1
  index_endpoints[nbins] = len(x)

  return index_endpoints


def _median_flatten_binned(x, y, nbins, x_aggr, y_aggr, bin_method):
  """Flattens by linearly interpolating between the median binned value."""
  x_aggr_fn = _get_aggr_fn(x_aggr)
  y_aggr_fn = _get_aggr_fn(y_aggr)

  if bin_method == "npts":
    endpoints = np.linspace(0, len(x), num=nbins + 1, dtype=np.int)
  elif bin_method == "xaxis":
    endpoints = _linear_bin_endpoints(x, nbins)
  else:
    raise ValueError("Unrecognized bin_method: %s" % bin_method)

  # The "xaxis" method of binning may result in some empty bins. Remove them.
  filtered_endpoints = []
  for index in endpoints:
    if not filtered_endpoints or index > filtered_endpoints[-1]:
      filtered_endpoints.append(index)
  endpoints = filtered_endpoints
  nbins = len(endpoints) - 1  # The last endpoint is the end of the last bin.

  binned_x = np.zeros(nbins, dtype=np.float32)
  binned_y = np.zeros(nbins, dtype=np.float32)

  for bin_index in range(nbins):
    start = endpoints[bin_index]
    end = endpoints[bin_index + 1]
    binned_x[bin_index] = x_aggr_fn(x[start:end])
    binned_y[bin_index] = y_aggr_fn(y[start:end])

  return y - np.interp(x, binned_x, binned_y)


def _median_filter_in_x(x, y, window_size):
  """A median filter whose bins have a fixed window_size on the x-axis."""
  assert len(x) == len(y)
  assert window_size > 0
  bin_start = 0
  bin_end = 0
  result = np.zeros_like(y)
  for i, bin_mid in enumerate(x):
    bin_min = bin_mid - window_size
    bin_max = bin_mid + window_size
    while x[bin_start] < bin_min:
      # Guaranteed to terminate with bin_start < len(x) since bin_min < x
      bin_start += 1
    while bin_end < len(x) and x[bin_end] < bin_max:
      bin_end += 1
    result[i] = np.median(y[bin_start:bin_end])
  return result


class BlsScorer(object):
  """Computes scores for a periodogram in different ways."""

  def __init__(self, results, ignore_negative_depth):
    self.results = results
    self.ignore_negative_depth = ignore_negative_depth

    self._raw_powers = None
    self._normalized_powers = None
    self._periods = None

  def score(self, method_name, **kwargs):
    scores = getattr(self, method_name)(**kwargs)
    return self._choose_top_result(scores)

  @property
  def raw_powers(self):
    if self._raw_powers is None:
      self._raw_powers = np.array(
          [result.bls_result.power for result in self.results])
    return self._raw_powers

  @property
  def normalized_powers(self):
    if self._normalized_powers is None:
      self._normalized_powers = [
          result.bls_result.power / np.log10(result.nbins)
          for result in self.results
      ]
    return self._normalized_powers

  @property
  def periods(self):
    if self._periods is None:
      self._periods = np.array([result.period for result in self.results])
    return self._periods

  def _choose_top_result(self, scores):
    if self.ignore_negative_depth:
      depths = [result.depth > 0 for result in self.results]
      scores *= np.greater(depths, 0, dtype=np.float)

    i = np.argmax(scores)
    return scores[i], self.results[i]

  def power(self, sqrt_power=False, normalize_by_bls_nbins=False):
    if normalize_by_bls_nbins:
      powers = self.normalized_powers
    else:
      powers = self.raw_powers

    if sqrt_power:
      powers = np.sqrt(powers)

    return powers

  def median_flattened(self,
                       nbins=10000,
                       period_scale="linear",
                       x_aggr="median",
                       y_aggr="median",
                       bin_method="npts",
                       sqrt_power=False,
                       normalize_by_bls_nbins=False):
    """Computes scores by flattening using linearly interpolated medians."""
    periods = self.periods
    if period_scale == "log":
      periods = np.log10(periods)
    elif period_scale == "inv":
      periods = 1.0 / self.periods
    elif period_scale != "linear":
      raise ValueError("Unexpected period_scale: %s" % period_scale)

    powers = self.power(sqrt_power, normalize_by_bls_nbins)
    flat_powers = _median_flatten_binned(
        periods,
        powers,
        nbins=nbins,
        x_aggr=x_aggr,
        y_aggr=y_aggr,
        bin_method=bin_method)

    mad = np.median(np.abs(flat_powers - np.median(flat_powers)))
    sn_bls = flat_powers / (mad / 0.67)

    return sn_bls

  def scatter_normalized(self,
                         window_size,
                         sqrt_power=False,
                         normalize_by_bls_nbins=False):
    """Computes scores by normalizing by the scatter."""
    powers = self.power(sqrt_power, normalize_by_bls_nbins)
    scatter = np.abs(np.diff(powers))

    # Diff returns a vector one less than the length of powers. We prepend the
    # first diff to make the sizes match.
    scatter = np.concatenate([[scatter[0]], scatter])

    scatter_trend = scipy.signal.medfilt(scatter, window_size)
    scores = powers / scatter_trend

    return scores

  def median_filter_normalized(self,
                               window_size,
                               divide=False,
                               sqrt_power=False,
                               normalize_by_bls_nbins=False,
                               normalize_by_mad=False):
    """Computes scores by flattening using a median filter."""
    powers = self.power(sqrt_power, normalize_by_bls_nbins)
    trend = scipy.signal.medfilt(powers, window_size)
    if divide:
      scores = powers / trend
    else:
      scores = powers - trend

    if normalize_by_mad:
      mad = np.median(np.abs(scores - np.median(scores)))
      scores /= (mad / 0.67)

    return scores

  def ofir(self,
           window_size,
           scatter_after_detrend=False,
           sqrt_power=False,
           normalize_by_bls_nbins=False):
    """Computes scores using the method of Ofir et al."""
    powers = self.power(sqrt_power, normalize_by_bls_nbins)

    trend = scipy.signal.medfilt(powers, window_size)
    scores = powers - trend

    if scatter_after_detrend:
      scatter = np.abs(np.diff(scores))
    else:
      scatter = np.abs(np.diff(powers))

    # Diff returns a vector one less than the length of powers. We prepend the
    # first diff to make the sizes match.
    scatter = np.concatenate([[scatter[0]], scatter])

    scatter_trend = scipy.signal.medfilt(scatter, window_size)
    scores /= scatter_trend

    return scores

  def sde(self, normalize_by_bls_nbins=False):
    """Computes scores using Signal Detection Efficiency."""
    signal_residues = self.power(True, normalize_by_bls_nbins)
    sr_mean = np.mean(signal_residues)
    sr_std = np.std(signal_residues)
    sdes = (signal_residues - sr_mean) / sr_std

    return sdes
