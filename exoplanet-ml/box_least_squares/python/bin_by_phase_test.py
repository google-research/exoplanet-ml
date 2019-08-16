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

"""Tests the Python wrapping of bin_by_phase.h."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from box_least_squares.python import bin_by_phase


class BinByPhaseTest(absltest.TestCase):

  def testEmptyTimeVector(self):
    with self.assertRaises(ValueError):
      bin_by_phase.bin_by_phase(time=[], values=[1, 2, 3], period=1, nbins=5)

  def testDifferentSizeVectors(self):
    with self.assertRaises(ValueError):
      bin_by_phase.bin_by_phase(
          time=[1, 2, 3, 4], values=[11, 22, 33, 44, 55], period=1, nbins=5)

  def testNonPositivePeriod(self):
    with self.assertRaises(ValueError):
      bin_by_phase.bin_by_phase(
          time=[1, 2, 3, 4, 5], values=[11, 22, 33, 44, 55], period=0, nbins=5)

  def testNonPositiveNbins(self):
    with self.assertRaises(ValueError):
      bin_by_phase.bin_by_phase(
          time=[1, 2, 3, 4, 5], values=[11, 22, 33, 44, 55], period=0, nbins=-1)

  def testBinByPhase(self):
    time = np.array(
        [
            0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
            1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4
        ],
        dtype=np.float)
    values = np.array(
        [
            -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10, 11, 12
        ],
        dtype=np.float)

    # If bin width (period / nbins) exactly divides (t mod period) then the
    # bin index may be one less than expected due to floating point arithmetic.
    # To ensure the bin indices are computed as expected, add a small offset to
    # the time values.
    time += 1e-10

    # Period larger than time range.
    binned_values, binned_square_values, bin_counts = bin_by_phase.bin_by_phase(
        time, values, period=5, nbins=5)
    np.testing.assert_almost_equal(binned_values, [-75, 25, 50, 0, 0])
    np.testing.assert_almost_equal(binned_square_values, [645, 145, 510, 0, 0])
    np.testing.assert_almost_equal(bin_counts, [10, 10, 5, 0, 0])

    # Period equal to time range.
    binned_values, binned_square_values, bin_counts = bin_by_phase.bin_by_phase(
        time, values, period=2.5, nbins=5)
    np.testing.assert_almost_equal(binned_values, [-50, -25, 0, 25, 50])
    np.testing.assert_almost_equal(binned_square_values,
                                   [510, 135, 10, 135, 510])
    np.testing.assert_almost_equal(bin_counts, [5, 5, 5, 5, 5])

    # Period smaller than time range.
    binned_values, binned_square_values, bin_counts = bin_by_phase.bin_by_phase(
        time, values, period=0.5, nbins=5)
    np.testing.assert_almost_equal(binned_values, [-10, -5, 0, 5, 10])
    np.testing.assert_almost_equal(binned_square_values,
                                   [270, 255, 250, 255, 270])
    np.testing.assert_almost_equal(bin_counts, [5, 5, 5, 5, 5])

    # All values in the first or third bin.
    binned_values, binned_square_values, bin_counts = bin_by_phase.bin_by_phase(
        time, values, period=0.2, nbins=5)
    np.testing.assert_almost_equal(binned_values, [0, 0, 0, 0, 0])
    np.testing.assert_almost_equal(binned_square_values, [728, 0, 572, 0, 0])
    np.testing.assert_almost_equal(bin_counts, [13, 0, 12, 0, 0])

    # All values in the first bin.
    binned_values, binned_square_values, bin_counts = bin_by_phase.bin_by_phase(
        time, values, period=0.1, nbins=5)
    np.testing.assert_almost_equal(binned_values, [0, 0, 0, 0, 0])
    np.testing.assert_almost_equal(binned_square_values, [1300, 0, 0, 0, 0])
    np.testing.assert_almost_equal(bin_counts, [25, 0, 0, 0, 0])

    # Values distributed.
    binned_values, binned_square_values, bin_counts = bin_by_phase.bin_by_phase(
        time, values, period=0.14159, nbins=5)
    np.testing.assert_almost_equal(binned_values, [-9, 6, 3, -10, 10])
    np.testing.assert_almost_equal(binned_square_values,
                                   [319, 182, 307, 174, 318])
    np.testing.assert_almost_equal(bin_counts, [6, 4, 6, 4, 5])


if __name__ == "__main__":
  absltest.main()
