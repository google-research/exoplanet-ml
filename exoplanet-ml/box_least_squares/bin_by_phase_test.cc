/* Copyright 2018 The Exoplanet ML Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "box_least_squares/bin_by_phase.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "box_least_squares/test_util.h"

using std::vector;

namespace exoplanet_ml {
namespace box_least_squares {
namespace {

class BinByPhaseTest : public ::testing::Test {
 protected:
  // Tests that BinByPhase() returns the expected error.
  void ExpectError(const vector<double>& time, const vector<double>& values,
                   const double period, const int nbins,
                   const std::string& expected_error) {
    EXPECT_FALSE(BinByPhase(time, values, period, nbins, &binned_values_,
                            &binned_square_values_, &bin_counts_, &error_));
    EXPECT_EQ(error_, expected_error);
  }

  // Tests that BinByPhase() produces expected outputs on specific inputs.
  void RunTest(const double period, const int nbins,
               const vector<double>& expected_values,
               const vector<double>& expected_square_values,
               const vector<int>& expected_counts) {
    // Input time series.
    vector<double> time = {0,   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                           0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
                           1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
    vector<double> values = {-12, -11, -10, -9, -8, -7, -6, -5, -4,
                             -3,  -2,  -1,  0,  1,  2,  3,  4,  5,
                             6,   7,   8,   9,  10, 11, 12};

    // If bin width (period / nbins) exactly divides (t mod period) then the
    // bin index may be one less than expected due to floating point arithmetic.
    // To ensure the bin indices are computed as expected, add a small offset to
    // the time values.
    for (double& t : time) t += 1e-10;

    ASSERT_TRUE(BinByPhase(time, values, period, nbins, &binned_values_,
                           &binned_square_values_, &bin_counts_, &error_));
    EXPECT_TRUE(error_.empty());
    EXPECT_THAT(binned_values_, testing::Pointwise(testing::DoubleNear(1e-12),
                                                   expected_values));
    EXPECT_THAT(
        binned_square_values_,
        testing::Pointwise(testing::DoubleNear(1e-12), expected_square_values));
    EXPECT_THAT(bin_counts_, testing::ElementsAreArray(expected_counts));
  }

  // Output arguments.
  vector<double> binned_values_;
  vector<double> binned_square_values_;
  vector<int> bin_counts_;
  std::string error_;
};

TEST_F(BinByPhaseTest, EmptyTimeVector) {
  vector<double> time = {};
  vector<double> values = {1, 2, 3};
  double period = 1;
  int nbins = 5;
  ExpectError(time, values, period, nbins, "time must not be empty");
}

TEST_F(BinByPhaseTest, DifferentSizeVectors) {
  vector<double> time = {1, 2, 3, 4};
  vector<double> values = {11, 22, 33, 44, 55};
  double period = 1;
  int nbins = 5;
  ExpectError(time, values, period, nbins,
              "time.size() (got: 4) != values.size() (got: 5)");
}

TEST_F(BinByPhaseTest, NonPositivePeriod) {
  vector<double> time = {1, 2, 3, 4, 5};
  vector<double> values = {11, 22, 33, 44, 55};
  double period = 0;
  int nbins = 5;
  ExpectError(time, values, period, nbins, "period must be positive (got: 0)");
}

TEST_F(BinByPhaseTest, NonPositiveNbins) {
  vector<double> time = {1, 2, 3, 4, 5};
  vector<double> values = {11, 22, 33, 44, 55};
  double period = 1;
  int nbins = -1;
  ExpectError(time, values, period, nbins, "nbins must be positive (got: -1)");
}

TEST_F(BinByPhaseTest, LargePeriod) {
  double period = 5;
  int nbins = 5;
  vector<double> expected_values = {-75, 25, 50, 0, 0};
  vector<double> expected_square_values = {645, 145, 510, 0, 0};
  vector<int> expected_counts = {10, 10, 5, 0, 0};
  RunTest(period, nbins, expected_values, expected_square_values,
          expected_counts);
}

TEST_F(BinByPhaseTest, PeriodEqualsTimeRange) {
  double period = 2.5;
  int nbins = 5;
  vector<double> expected_values = {-50, -25, 0, 25, 50};
  vector<double> expected_square_values = {510, 135, 10, 135, 510};
  vector<int> expected_counts = {5, 5, 5, 5, 5};
  RunTest(period, nbins, expected_values, expected_square_values,
          expected_counts);
}

TEST_F(BinByPhaseTest, SmallPeriod) {
  double period = 0.5;
  int nbins = 5;
  vector<double> expected_values = {-10, -5, 0, 5, 10};
  vector<double> expected_square_values = {270, 255, 250, 255, 270};
  vector<int> expected_counts = {5, 5, 5, 5, 5};
  RunTest(period, nbins, expected_values, expected_square_values,
          expected_counts);
}

TEST_F(BinByPhaseTest, FirstAndThirdBinsOnly) {
  double period = 0.2;
  int nbins = 5;
  vector<double> expected_values = {0, 0, 0, 0, 0};
  vector<double> expected_square_values = {728, 0, 572, 0, 0};
  vector<int> expected_counts = {13, 0, 12, 0, 0};
  RunTest(period, nbins, expected_values, expected_square_values,
          expected_counts);
}

TEST_F(BinByPhaseTest, FirstBinOnly) {
  double period = 0.1;
  int nbins = 5;
  vector<double> expected_values = {0, 0, 0, 0, 0};
  vector<double> expected_square_values = {1300, 0, 0, 0, 0};
  vector<int> expected_counts = {25, 0, 0, 0, 0};
  RunTest(period, nbins, expected_values, expected_square_values,
          expected_counts);
}

TEST_F(BinByPhaseTest, ValuesDistributed) {
  double period = 0.14159;
  int nbins = 5;
  vector<double> expected_values = {-9, 6, 3, -10, 10};
  vector<double> expected_square_values = {319, 182, 307, 174, 318};
  vector<int> expected_counts = {6, 4, 6, 4, 5};
  RunTest(period, nbins, expected_values, expected_square_values,
          expected_counts);
}

TEST_F(BinByPhaseTest, VariableBinSize) {
  double period = 1;

  // Start out with 5 bins.
  int nbins = 5;
  vector<double> expected_values = {-9, 3, 2, -2, 6};
  vector<double> expected_square_values = {415, 403, 270, 102, 110};
  vector<int> expected_counts = {6, 6, 5, 4, 4};
  RunTest(period, nbins, expected_values, expected_square_values,
          expected_counts);

  // Expand to 10 bins.
  nbins = 10;
  expected_values = {-6, -3, 0, 3, 6, -4, -2, 0, 2, 4};
  expected_square_values = {212, 203, 200, 203, 212, 58, 52, 50, 52, 58};
  expected_counts = {3, 3, 3, 3, 3, 2, 2, 2, 2, 2};
  RunTest(period, nbins, expected_values, expected_square_values,
          expected_counts);

  // Shrink to 4 bins.
  nbins = 4;
  expected_values = {-9, 9, -6, 6};
  expected_square_values = {615, 415, 160, 110};
  expected_counts = {9, 6, 6, 4};
  RunTest(period, nbins, expected_values, expected_square_values,
          expected_counts);
}

}  // namespace
}  // namespace box_least_squares
}  // namespace exoplanet_ml
