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

"""Tests the Python wrapping of box_least_squares.h."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from box_least_squares import box_least_squares_pb2 as bls_pb2
from box_least_squares.python import box_least_squares


class RunBlsTest(absltest.TestCase):

  def testRunBls(self):
    values = [-30, 70, 70, 70, -30, -30, -30, -30, -30, -30]
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    total_signal = 2100

    # Narrow box.
    options = bls_pb2.BlsOptions()
    result = box_least_squares.run_bls(values, weights, options)
    self.assertEqual(result.start, 1)
    self.assertEqual(result.width, 3)
    self.assertAlmostEqual(result.r, 0.3)
    self.assertAlmostEqual(result.s, 21)
    self.assertAlmostEqual(result.t, 1470)
    self.assertAlmostEqual(result.total_signal, total_signal)
    self.assertAlmostEqual(result.power, total_signal)  # Perfect fit.
    self.assertAlmostEqual(result.mse, 0)  # Perfect fit.

    # Wide box.
    options.width_min = 4
    options.width_max = 9
    result = box_least_squares.run_bls(values, weights, options)
    self.assertEqual(result.start, 4)
    self.assertEqual(result.width, 7)
    self.assertAlmostEqual(result.r, 0.7)
    self.assertAlmostEqual(result.s, -21)
    self.assertAlmostEqual(result.t, 630)
    self.assertAlmostEqual(result.total_signal, total_signal)
    self.assertAlmostEqual(result.power, total_signal)  # Perfect fit.
    self.assertAlmostEqual(result.mse, 0)  # Perfect fit.


class BoxLeastSquresTest(absltest.TestCase):

  def testBoxLeastSquares(self):
    # Create a time series evenly spaced at time intervals of width 10, starting
    # at 3, which periodically repeats the sequence [-2, -2, -2, 3, 3, 3, 3, -2,
    # -2, -2]. Its period is therefore 100.0.
    time = np.arange(0, 1000, 10) + 3
    values = np.where(
        np.logical_and(
            np.arange(100, dtype=np.int) % 10 >= 3,
            np.arange(100, dtype=np.int) % 10 < 7), 3, -2)

    bls = box_least_squares.BoxLeastSquares(time, values, capacity=20)

    # 20 bins.
    nbins = 20
    options = bls_pb2.BlsOptions(width_max=10)
    best_result = None
    for period in range(10, 201, 10):
      result = bls.fit(period, nbins, options)
      if (best_result is None or
          result.bls_result.power > best_result.bls_result.power):
        best_result = result

    self.assertLen(bls.binned_weighted_values, 20)
    self.assertLen(bls.binned_weighted_square_values, 20)
    self.assertLen(bls.binned_weights, 20)

    self.assertEqual(best_result.nbins, 20)
    self.assertEqual(best_result.bls_result.start, 6)
    self.assertEqual(best_result.bls_result.width, 7)
    self.assertAlmostEqual(best_result.bls_result.r, 0.4)
    self.assertAlmostEqual(best_result.bls_result.s, 1.2)
    self.assertAlmostEqual(best_result.bls_result.t, 3.6, 6)
    self.assertAlmostEqual(best_result.bls_result.total_signal, 6)
    self.assertAlmostEqual(best_result.bls_result.mse, 0)
    self.assertAlmostEqual(best_result.period, 100)
    self.assertAlmostEqual(best_result.duration, 35)
    self.assertAlmostEqual(best_result.epoch, 47.5)
    self.assertAlmostEqual(best_result.depth, -5)
    self.assertAlmostEqual(best_result.baseline, -2)

    # 10 bins.
    nbins = 10
    options = bls_pb2.BlsOptions(width_max=5)
    best_result = None
    for period in range(10, 201, 10):
      result = bls.fit(period, nbins, options)
      if (best_result is None or
          result.bls_result.power > best_result.bls_result.power):
        best_result = result

    self.assertLen(bls.binned_weighted_values, 10)
    self.assertLen(bls.binned_weighted_square_values, 10)
    self.assertLen(bls.binned_weights, 10)

    self.assertEqual(best_result.nbins, 10)
    self.assertEqual(best_result.bls_result.start, 3)
    self.assertEqual(best_result.bls_result.width, 4)
    self.assertAlmostEqual(best_result.bls_result.r, 0.4)
    self.assertAlmostEqual(best_result.bls_result.s, 1.2)
    self.assertAlmostEqual(best_result.bls_result.t, 3.6, 6)
    self.assertAlmostEqual(best_result.bls_result.total_signal, 6)
    self.assertAlmostEqual(best_result.bls_result.mse, 0)
    self.assertAlmostEqual(best_result.period, 100)
    self.assertAlmostEqual(best_result.duration, 40)
    self.assertAlmostEqual(best_result.epoch, 50)
    self.assertAlmostEqual(best_result.depth, -5)
    self.assertAlmostEqual(best_result.baseline, -2)

    # 25 bins.
    nbins = 25
    options = bls_pb2.BlsOptions(width_max=12)
    best_result = None
    for period in range(10, 201, 10):
      result = bls.fit(period, nbins, options)
      if (best_result is None or
          result.bls_result.power > best_result.bls_result.power):
        best_result = result

    self.assertLen(bls.binned_weighted_values, 25)
    self.assertLen(bls.binned_weighted_square_values, 25)
    self.assertLen(bls.binned_weights, 25)

    self.assertEqual(best_result.nbins, 25)
    self.assertEqual(best_result.bls_result.start, 8)
    self.assertEqual(best_result.bls_result.width, 8)
    self.assertAlmostEqual(best_result.bls_result.r, 0.4)
    self.assertAlmostEqual(best_result.bls_result.s, 1.2)
    self.assertAlmostEqual(best_result.bls_result.t, 3.6, 6)
    self.assertAlmostEqual(best_result.bls_result.total_signal, 6)
    self.assertAlmostEqual(best_result.bls_result.mse, 0)
    self.assertAlmostEqual(best_result.period, 100)
    self.assertAlmostEqual(best_result.duration, 32)
    self.assertAlmostEqual(best_result.epoch, 48)
    self.assertAlmostEqual(best_result.depth, -5)
    self.assertAlmostEqual(best_result.baseline, -2)


if __name__ == "__main__":
  absltest.main()
