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

#include "box_least_squares/box_least_squares.h"

#include "gtest/gtest.h"
#include "box_least_squares/box_least_squares.pb.h"
#include "box_least_squares/test_util.h"
#include "box_least_squares/util.h"

using std::vector;

namespace exoplanet_ml {
namespace box_least_squares {
namespace {

class RunBlsTest : public ::testing::Test {
 protected:
  // Output arguments.
  BlsResult result_;
  string error_;
};

TEST_F(RunBlsTest, PerfectFit1) {
  vector<double> values = {-30, 70, 70, 70, -30, -30, -30, -30, -30, -30};
  vector<double> weights = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  double total_signal = 2100;

  // Narrow box.
  BlsOptions options;
  ASSERT_TRUE(RunBls(values, weights, options, &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 1);
  EXPECT_EQ(result_.width(), 3);
  EXPECT_FLOAT_EQ(result_.r(), 0.3);
  EXPECT_FLOAT_EQ(result_.s(), 21);
  EXPECT_FLOAT_EQ(result_.t(), 1470);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_FLOAT_EQ(result_.power(), total_signal);  // Perfect fit.
  EXPECT_FLOAT_EQ(result_.mse(), 0);               // Perfect fit.

  // Wide box.
  options.set_width_min(4);
  options.set_width_max(9);
  ASSERT_TRUE(RunBls(values, weights, options, &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 4);
  EXPECT_EQ(result_.width(), 7);
  EXPECT_FLOAT_EQ(result_.r(), 0.7);
  EXPECT_FLOAT_EQ(result_.s(), -21);
  EXPECT_FLOAT_EQ(result_.t(), 630);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_FLOAT_EQ(result_.power(), total_signal);  // Perfect fit.
  EXPECT_FLOAT_EQ(result_.mse(), 0);               // Perfect fit.
}

TEST_F(RunBlsTest, PerfectFit2) {
  vector<double> values = {20, 20, 20, 20, 20, -30, -30, -30, -30, 20};
  vector<double> weights = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
  double total_signal = 600;

  // Narrow box.
  BlsOptions options;
  ASSERT_TRUE(RunBls(values, weights, options, &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 5);
  EXPECT_EQ(result_.width(), 4);
  EXPECT_FLOAT_EQ(result_.r(), 0.4);
  EXPECT_FLOAT_EQ(result_.s(), -12);
  EXPECT_FLOAT_EQ(result_.t(), 360);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_FLOAT_EQ(result_.power(), total_signal);  // Perfect fit.
  EXPECT_FLOAT_EQ(result_.mse(), 0);               // Perfect fit.

  // Wide box.
  options.set_width_min(5);
  options.set_width_max(9);
  ASSERT_TRUE(RunBls(values, weights, options, &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 9);
  EXPECT_EQ(result_.width(), 6);
  EXPECT_FLOAT_EQ(result_.r(), 0.6);
  EXPECT_FLOAT_EQ(result_.s(), 12);
  EXPECT_FLOAT_EQ(result_.t(), 240);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_FLOAT_EQ(result_.power(), total_signal);  // Perfect fit.
  EXPECT_FLOAT_EQ(result_.mse(), 0);               // Perfect fit.
}

TEST(BoxLeastSquaresTest, FitCentered) {
  // Create a time series evenly spaced at time intervals of width 10, starting
  // at 3, which periodically repeats the sequence {-2, -2, -2, 3, 3, 3, 3, -2,
  // -2, -2}. Its period is therefore 100.0.
  const int npts = 100;
  vector<double> time(npts);
  vector<double> values(npts);
  for (int i = 0; i < npts; ++i) {
    time[i] = i * 10 + 3;
    values[i] = (i % 10) >= 3 && (i % 10) < 7 ? 3 : -2;
  }

  // Initialize a BoxLeastSquares that can be used for different periods and
  // different numbers of bins. Set the initial capacity of the binning vectors
  // to 20.
  BoxLeastSquares bls(time, values, 20);
  EXPECT_FLOAT_EQ(bls.get_mean_value(), 0);

  // Consider periods between 10 and 200.
  vector<double> periods = {10,  20,  30,  40,  50,  60,  70,  80,  90,  100,
                            110, 120, 130, 140, 150, 160, 170, 180, 190, 200};

  // nbins = 20.
  {
    // BLS options.
    int nbins = 20;
    BlsOptions options;

    // Output arguments.
    BoxTransitModel result;
    string error;

    // Run BLS on all periods.
    BoxTransitModel best_result;
    for (auto period : periods) {
      bls.Fit(period, nbins, options, &result, &error);
      EXPECT_TRUE(error.empty());
      EXPECT_EQ(bls.get_binned_weighted_values().size(), 20);
      ASSERT_NEAR(Sum(bls.get_binned_weighted_values()), 0, 1e-12);
      EXPECT_EQ(bls.get_binned_weighted_square_values().size(), 20);
      // binned_weighted_square_values = sum(square(values - E(values))) / npts
      ASSERT_NEAR(Sum(bls.get_binned_weighted_square_values()), 6, 1e-12);
      EXPECT_EQ(bls.get_binned_weights().size(), 20);
      ASSERT_NEAR(Sum(bls.get_binned_weights()), 1, 1e-12);
      if (result.bls_result().power() > best_result.bls_result().power()) {
        best_result = result;
      }
    }
    // For period = 100, the box spans bins 6 = [30, 35) to 12 = [60, 65).
    EXPECT_EQ(best_result.nbins(), 20);
    EXPECT_EQ(best_result.bls_result().start(), 6);
    EXPECT_EQ(best_result.bls_result().width(), 7);
    EXPECT_FLOAT_EQ(best_result.bls_result().r(), 0.4);
    EXPECT_FLOAT_EQ(best_result.bls_result().s(), 1.2);
    EXPECT_FLOAT_EQ(best_result.bls_result().t(), 3.6);
    EXPECT_FLOAT_EQ(best_result.bls_result().total_signal(), 6);
    EXPECT_NEAR(best_result.bls_result().mse(), 0, 1e-12);
    EXPECT_FLOAT_EQ(best_result.period(), 100);
    EXPECT_FLOAT_EQ(best_result.duration(), 35);
    EXPECT_FLOAT_EQ(best_result.epoch(), 47.5);
    EXPECT_FLOAT_EQ(best_result.depth(), -5);
    EXPECT_FLOAT_EQ(best_result.baseline(), -2);
  }

  // nbins = 10.
  {
    // BLS options.
    int nbins = 10;
    BlsOptions options;

    // Output arguments.
    BoxTransitModel result;
    string error;

    // Run BLS on all periods.
    BoxTransitModel best_result;
    for (auto period : periods) {
      bls.Fit(period, nbins, options, &result, &error);
      EXPECT_TRUE(error.empty());
      EXPECT_EQ(bls.get_binned_weighted_values().size(), 10);
      ASSERT_NEAR(Sum(bls.get_binned_weighted_values()), 0, 1e-12);
      EXPECT_EQ(bls.get_binned_weighted_square_values().size(), 10);
      // binned_weighted_square_values = sum(square(values - E(values))) / npts
      ASSERT_NEAR(Sum(bls.get_binned_weighted_square_values()), 6, 1e-12);
      EXPECT_EQ(bls.get_binned_weights().size(), 10);
      ASSERT_NEAR(Sum(bls.get_binned_weights()), 1, 1e-12);
      if (result.bls_result().power() > best_result.bls_result().power()) {
        best_result = result;
      }
    }
    // For period = 100, the box spans bins 3 = [30, 40) to 6 = [60, 70).
    EXPECT_EQ(best_result.nbins(), 10);
    EXPECT_EQ(best_result.bls_result().start(), 3);
    EXPECT_EQ(best_result.bls_result().width(), 4);
    EXPECT_FLOAT_EQ(best_result.bls_result().r(), 0.4);
    EXPECT_FLOAT_EQ(best_result.bls_result().s(), 1.2);
    EXPECT_FLOAT_EQ(best_result.bls_result().t(), 3.6);
    EXPECT_FLOAT_EQ(best_result.bls_result().total_signal(), 6);
    EXPECT_NEAR(best_result.bls_result().mse(), 0, 1e-12);
    EXPECT_FLOAT_EQ(best_result.period(), 100);
    EXPECT_FLOAT_EQ(best_result.duration(), 40);
    EXPECT_FLOAT_EQ(best_result.epoch(), 50);
    EXPECT_FLOAT_EQ(best_result.depth(), -5);
    EXPECT_FLOAT_EQ(best_result.baseline(), -2);
  }

  // nbins = 25.
  {
    // BLS options.
    int nbins = 25;
    BlsOptions options;

    // Output arguments.
    BoxTransitModel result;
    string error;

    // Run BLS on all periods.
    BoxTransitModel best_result;
    for (auto period : periods) {
      bls.Fit(period, nbins, options, &result, &error);
      EXPECT_TRUE(error.empty());
      EXPECT_EQ(bls.get_binned_weighted_values().size(), 25);
      ASSERT_NEAR(Sum(bls.get_binned_weighted_values()), 0, 1e-12);
      EXPECT_EQ(bls.get_binned_weighted_square_values().size(), 25);
      // binned_weighted_square_values = sum(square(values - E(values))) / npts
      ASSERT_NEAR(Sum(bls.get_binned_weighted_square_values()), 6, 1e-12);
      EXPECT_EQ(bls.get_binned_weights().size(), 25);
      ASSERT_NEAR(Sum(bls.get_binned_weights()), 1, 1e-12);
      if (result.bls_result().power() > best_result.bls_result().power()) {
        best_result = result;
      }
    }
    // For period = 100, the box spans bins 8 = [32, 36) to 15 = [60, 64).
    EXPECT_EQ(best_result.nbins(), 25);
    EXPECT_EQ(best_result.bls_result().start(), 8);
    EXPECT_EQ(best_result.bls_result().width(), 8);
    EXPECT_FLOAT_EQ(best_result.bls_result().r(), 0.4);
    EXPECT_FLOAT_EQ(best_result.bls_result().s(), 1.2);
    EXPECT_FLOAT_EQ(best_result.bls_result().t(), 3.6);
    EXPECT_FLOAT_EQ(best_result.bls_result().total_signal(), 6);
    EXPECT_NEAR(best_result.bls_result().mse(), 0, 1e-12);
    EXPECT_FLOAT_EQ(best_result.period(), 100);
    EXPECT_FLOAT_EQ(best_result.duration(), 32);
    EXPECT_FLOAT_EQ(best_result.epoch(), 48);
    EXPECT_FLOAT_EQ(best_result.depth(), -5);
    EXPECT_FLOAT_EQ(best_result.baseline(), -2);
  }
}

TEST(BoxLeastSquaresTest, FitEdgeAligned) {
  // Create a time series evenly spaced at time intervals of width 10, starting
  // at 3, which periodically repeats the sequence {2, 2, 2, 2, 2, 2, -3,
  // -3, -3, -3}. Its period is therefore 100.0.
  const int npts = 100;
  vector<double> time(npts);
  vector<double> values(npts);
  for (int i = 0; i < npts; ++i) {
    time[i] = i * 10 + 3;
    values[i] = (i % 10) < 6 ? 2 : -3;
  }

  // Initialize a BoxLeastSquares that can be used for different periods and
  // different numbers of bins. Set the initial capacity of the binning vectors
  // to 20.
  BoxLeastSquares bls(time, values, 20);
  EXPECT_FLOAT_EQ(bls.get_mean_value(), 0);

  // Consider periods between 10 and 200.
  vector<double> periods = {10,  20,  30,  40,  50,  60,  70,  80,  90,  100,
                            110, 120, 130, 140, 150, 160, 170, 180, 190, 200};

  // nbins = 20.
  {
    // BLS options.
    int nbins = 20;
    BlsOptions options;

    // Output arguments.
    BoxTransitModel result;
    string error;

    // Run BLS on all periods.
    BoxTransitModel best_result;
    for (auto period : periods) {
      bls.Fit(period, nbins, options, &result, &error);
      EXPECT_TRUE(error.empty());
      EXPECT_EQ(bls.get_binned_weighted_values().size(), 20);
      ASSERT_NEAR(Sum(bls.get_binned_weighted_values()), 0, 1e-12);
      EXPECT_EQ(bls.get_binned_weighted_square_values().size(), 20);
      // binned_weighted_square_values = sum(square(values - E(values))) / npts
      ASSERT_NEAR(Sum(bls.get_binned_weighted_square_values()), 6, 1e-12);
      EXPECT_EQ(bls.get_binned_weights().size(), 20);
      ASSERT_NEAR(Sum(bls.get_binned_weights()), 1, 1e-12);
      if (result.bls_result().power() > best_result.bls_result().power()) {
        best_result = result;
      }
    }
    // For period = 100, the box spans bins 12 = [60, 65) to 18 = [90, 95).
    EXPECT_EQ(best_result.nbins(), 20);
    EXPECT_EQ(best_result.bls_result().start(), 12);
    EXPECT_EQ(best_result.bls_result().width(), 7);
    EXPECT_FLOAT_EQ(best_result.bls_result().r(), 0.4);
    EXPECT_FLOAT_EQ(best_result.bls_result().s(), -1.2);
    EXPECT_FLOAT_EQ(best_result.bls_result().t(), 3.6);
    EXPECT_FLOAT_EQ(best_result.bls_result().total_signal(), 6);
    EXPECT_NEAR(best_result.bls_result().mse(), 0, 1e-12);
    EXPECT_FLOAT_EQ(best_result.period(), 100);
    EXPECT_FLOAT_EQ(best_result.duration(), 35);
    EXPECT_FLOAT_EQ(best_result.epoch(), 77.5);
    EXPECT_FLOAT_EQ(best_result.depth(), 5);
    EXPECT_FLOAT_EQ(best_result.baseline(), 2);
  }

  // nbins = 10.
  {
    // BLS options.
    int nbins = 10;
    BlsOptions options;

    // Output arguments.
    BoxTransitModel result;
    string error;

    // Run BLS on all periods.
    BoxTransitModel best_result;
    for (auto period : periods) {
      bls.Fit(period, nbins, options, &result, &error);
      EXPECT_TRUE(error.empty());
      EXPECT_EQ(bls.get_binned_weighted_values().size(), 10);
      ASSERT_NEAR(Sum(bls.get_binned_weighted_values()), 0, 1e-12);
      EXPECT_EQ(bls.get_binned_weighted_square_values().size(), 10);
      // binned_weighted_square_values = sum(square(values - E(values))) / npts
      ASSERT_NEAR(Sum(bls.get_binned_weighted_square_values()), 6, 1e-12);
      EXPECT_EQ(bls.get_binned_weights().size(), 10);
      ASSERT_NEAR(Sum(bls.get_binned_weights()), 1, 1e-12);
      if (result.bls_result().power() > best_result.bls_result().power()) {
        best_result = result;
      }
    }
    // For period = 100, the box spans bins 6 = [60, 70) to 9 = [90, 93).
    EXPECT_EQ(best_result.nbins(), 10);
    EXPECT_EQ(best_result.bls_result().start(), 6);
    EXPECT_EQ(best_result.bls_result().width(), 4);
    EXPECT_FLOAT_EQ(best_result.bls_result().r(), 0.4);
    EXPECT_FLOAT_EQ(best_result.bls_result().s(), -1.2);
    EXPECT_FLOAT_EQ(best_result.bls_result().t(), 3.6);
    EXPECT_FLOAT_EQ(best_result.bls_result().total_signal(), 6);
    EXPECT_NEAR(best_result.bls_result().mse(), 0, 1e-12);
    EXPECT_FLOAT_EQ(best_result.period(), 100);
    EXPECT_FLOAT_EQ(best_result.duration(), 40);
    EXPECT_FLOAT_EQ(best_result.epoch(), 80);
    EXPECT_FLOAT_EQ(best_result.depth(), 5);
    EXPECT_FLOAT_EQ(best_result.baseline(), 2);
  }

  // nbins = 25.
  {
    // BLS options.
    int nbins = 25;
    BlsOptions options;

    // Output arguments.
    BoxTransitModel result;
    string error;

    // Run BLS on all periods.
    BoxTransitModel best_result;
    for (auto period : periods) {
      bls.Fit(period, nbins, options, &result, &error);
      EXPECT_TRUE(error.empty());
      EXPECT_EQ(bls.get_binned_weighted_values().size(), 25);
      ASSERT_NEAR(Sum(bls.get_binned_weighted_values()), 0, 1e-12);
      EXPECT_EQ(bls.get_binned_weighted_square_values().size(), 25);
      // binned_weighted_square_values = sum(square(values - E(values))) / npts
      ASSERT_NEAR(Sum(bls.get_binned_weighted_square_values()), 6, 1e-12);
      EXPECT_EQ(bls.get_binned_weights().size(), 25);
      ASSERT_NEAR(Sum(bls.get_binned_weights()), 1, 1e-12);
      if (result.bls_result().power() > best_result.bls_result().power()) {
        best_result = result;
      }
    }
    // For period = 100, the box spans bins 15 = [60, 64) to 23 = [92, 96).
    EXPECT_EQ(best_result.nbins(), 25);
    EXPECT_EQ(best_result.bls_result().start(), 15);
    EXPECT_EQ(best_result.bls_result().width(), 9);
    EXPECT_FLOAT_EQ(best_result.bls_result().r(), 0.4);
    EXPECT_FLOAT_EQ(best_result.bls_result().s(), -1.2);
    EXPECT_FLOAT_EQ(best_result.bls_result().t(), 3.6);
    EXPECT_FLOAT_EQ(best_result.bls_result().total_signal(), 6);
    EXPECT_NEAR(best_result.bls_result().mse(), 0, 1e-12);
    EXPECT_FLOAT_EQ(best_result.period(), 100);
    EXPECT_FLOAT_EQ(best_result.duration(), 36);
    EXPECT_FLOAT_EQ(best_result.epoch(), 78);
    EXPECT_FLOAT_EQ(best_result.depth(), 5);
    EXPECT_FLOAT_EQ(best_result.baseline(), 2);
  }
}

TEST(BoxLeastSquaresTest, FitSplit) {
  // Create a time series evenly spaced at time intervals of width 10, starting
  // at 123, which periodically repeats the sequence {2, 2, 2, 2, 2, 2, -3, - 3,
  // -3, -3}. Its period is 100, but when phase folded, the box is split  across
  // the left and right edges.
  const int npts = 100;
  vector<double> time(npts);
  vector<double> values(npts);
  for (int i = 0; i < npts; ++i) {
    time[i] = i * 10 + 123;
    values[i] = (i % 10) < 6 ? 2 : -3;
  }

  // Initialize a BoxLeastSquares that can be used for different periods and
  // different numbers of bins. Set the initial capacity of the binning vectors
  // to 20.
  BoxLeastSquares bls(time, values, 20);
  EXPECT_FLOAT_EQ(bls.get_mean_value(), 0);

  // Consider periods between 10 and 200.
  vector<double> periods = {10,  20,  30,  40,  50,  60,  70,  80,  90,  100,
                            110, 120, 130, 140, 150, 160, 170, 180, 190, 200};

  // nbins = 20.
  {
    // BLS options.
    int nbins = 20;
    BlsOptions options;

    // Output arguments.
    BoxTransitModel result;
    string error;

    // Run BLS on all periods.
    BoxTransitModel best_result;
    for (auto period : periods) {
      bls.Fit(period, nbins, options, &result, &error);
      EXPECT_TRUE(error.empty());
      EXPECT_EQ(bls.get_binned_weighted_values().size(), 20);
      ASSERT_NEAR(Sum(bls.get_binned_weighted_values()), 0, 1e-12);
      EXPECT_EQ(bls.get_binned_weighted_square_values().size(), 20);
      // binned_weighted_square_values = sum(square(values - E(values))) / npts
      ASSERT_NEAR(Sum(bls.get_binned_weighted_square_values()), 6, 1e-12);
      EXPECT_EQ(bls.get_binned_weights().size(), 20);
      ASSERT_NEAR(Sum(bls.get_binned_weights()), 1, 1e-12);
      if (result.bls_result().power() > best_result.bls_result().power()) {
        best_result = result;
      }
    }
    // For period = 100, the box spans [80, 85) (bin 16) to [110, 115) =
    // [10, 15) (bin 2).
    EXPECT_EQ(best_result.nbins(), 20);
    EXPECT_EQ(best_result.bls_result().start(), 16);
    EXPECT_EQ(best_result.bls_result().width(), 7);
    EXPECT_FLOAT_EQ(best_result.bls_result().r(), 0.4);
    EXPECT_FLOAT_EQ(best_result.bls_result().s(), -1.2);
    EXPECT_FLOAT_EQ(best_result.bls_result().t(), 3.6);
    EXPECT_FLOAT_EQ(best_result.bls_result().total_signal(), 6);
    EXPECT_NEAR(best_result.bls_result().mse(), 0, 1e-12);
    EXPECT_FLOAT_EQ(best_result.period(), 100);
    EXPECT_FLOAT_EQ(best_result.duration(), 35);
    EXPECT_FLOAT_EQ(best_result.epoch(), 97.5);
    EXPECT_FLOAT_EQ(best_result.depth(), 5);
    EXPECT_FLOAT_EQ(best_result.baseline(), 2);
  }

  // nbins = 10.
  {
    // BLS options.
    int nbins = 10;
    BlsOptions options;

    // Output arguments.
    BoxTransitModel result;
    string error;

    // Run BLS on all periods.
    BoxTransitModel best_result;
    for (auto period : periods) {
      bls.Fit(period, nbins, options, &result, &error);
      EXPECT_TRUE(error.empty());
      EXPECT_EQ(bls.get_binned_weighted_values().size(), 10);
      ASSERT_NEAR(Sum(bls.get_binned_weighted_values()), 0, 1e-12);
      EXPECT_EQ(bls.get_binned_weighted_square_values().size(), 10);
      // binned_weighted_square_values = sum(square(values - E(values))) / npts
      ASSERT_NEAR(Sum(bls.get_binned_weighted_square_values()), 6, 1e-12);
      EXPECT_EQ(bls.get_binned_weights().size(), 10);
      ASSERT_NEAR(Sum(bls.get_binned_weights()), 1, 1e-12);
      if (result.bls_result().power() > best_result.bls_result().power()) {
        best_result = result;
      }
    }
    // For period = 100, the box spans [80, 90) (bin 8) to [110, 120) =
    // [10, 20) (bin 1).
    EXPECT_EQ(best_result.nbins(), 10);
    EXPECT_EQ(best_result.bls_result().start(), 8);
    EXPECT_EQ(best_result.bls_result().width(), 4);
    EXPECT_FLOAT_EQ(best_result.bls_result().r(), 0.4);
    EXPECT_FLOAT_EQ(best_result.bls_result().s(), -1.2);
    EXPECT_FLOAT_EQ(best_result.bls_result().t(), 3.6);
    EXPECT_FLOAT_EQ(best_result.bls_result().total_signal(), 6);
    EXPECT_NEAR(best_result.bls_result().mse(), 0, 1e-12);
    EXPECT_FLOAT_EQ(best_result.period(), 100);
    EXPECT_FLOAT_EQ(best_result.duration(), 40);
    EXPECT_FLOAT_EQ(best_result.epoch(), 0);
    EXPECT_FLOAT_EQ(best_result.depth(), 5);
    EXPECT_FLOAT_EQ(best_result.baseline(), 2);
  }

  // nbins = 25.
  {
    // BLS options.
    int nbins = 25;
    BlsOptions options;

    // Output arguments.
    BoxTransitModel result;
    string error;

    // Run BLS on all periods.
    BoxTransitModel best_result;
    for (auto period : periods) {
      bls.Fit(period, nbins, options, &result, &error);
      EXPECT_TRUE(error.empty());
      EXPECT_EQ(bls.get_binned_weighted_values().size(), 25);
      ASSERT_NEAR(Sum(bls.get_binned_weighted_values()), 0, 1e-12);
      EXPECT_EQ(bls.get_binned_weighted_square_values().size(), 25);
      // binned_weighted_square_values = sum(square(values - E(values))) / npts
      ASSERT_NEAR(Sum(bls.get_binned_weighted_square_values()), 6, 1e-12);
      EXPECT_EQ(bls.get_binned_weights().size(), 25);
      ASSERT_NEAR(Sum(bls.get_binned_weights()), 1, 1e-12);
      if (result.bls_result().power() > best_result.bls_result().power()) {
        best_result = result;
      }
    }
    // For period = 100, the box spans [80, 84) (bin 20) to [112, 116) =
    // [12, 16) (bin 3).
    EXPECT_EQ(best_result.nbins(), 25);
    EXPECT_EQ(best_result.bls_result().start(), 20);
    EXPECT_EQ(best_result.bls_result().width(), 9);
    EXPECT_FLOAT_EQ(best_result.bls_result().r(), 0.4);
    EXPECT_FLOAT_EQ(best_result.bls_result().s(), -1.2);
    EXPECT_FLOAT_EQ(best_result.bls_result().t(), 3.6);
    EXPECT_FLOAT_EQ(best_result.bls_result().total_signal(), 6);
    EXPECT_NEAR(best_result.bls_result().mse(), 0, 1e-12);
    EXPECT_FLOAT_EQ(best_result.period(), 100);
    EXPECT_FLOAT_EQ(best_result.duration(), 36);
    EXPECT_FLOAT_EQ(best_result.epoch(), 98);
    EXPECT_FLOAT_EQ(best_result.depth(), 5);
    EXPECT_FLOAT_EQ(best_result.baseline(), 2);
  }
}

TEST(BoxLeastSquaresTest, FitPositiveMean) {
  // Create a time series evenly spaced at time intervals of width 10, starting
  // at 3, which periodically repeats the sequence {20, 20, 20, 10, 10, 10, 10,
  // 20, 20, 20}. Its period is therefore 100.0.
  const int npts = 100;
  vector<double> time(npts);
  vector<double> values(npts);
  for (int i = 0; i < npts; ++i) {
    time[i] = i * 10 + 3;
    values[i] = (i % 10) >= 3 && (i % 10) < 7 ? 10 : 20;
  }

  // Initialize a BoxLeastSquares that can be used for different periods and
  // different numbers of bins. Set the initial capacity of the binning vectors
  // to 20.
  BoxLeastSquares bls(time, values, 20);
  EXPECT_FLOAT_EQ(bls.get_mean_value(), 16);

  // Consider periods between 10 and 200.
  vector<double> periods = {10,  20,  30,  40,  50,  60,  70,  80,  90,  100,
                            110, 120, 130, 140, 150, 160, 170, 180, 190, 200};

  // nbins = 10.
  // BLS options.
  int nbins = 10;
  BlsOptions options;

  // Output arguments.
  BoxTransitModel result;
  string error;

  // Run BLS on all periods.
  BoxTransitModel best_result;
  for (auto period : periods) {
    bls.Fit(period, nbins, options, &result, &error);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(bls.get_binned_weighted_values().size(), 10);
    ASSERT_NEAR(Sum(bls.get_binned_weighted_values()), 0, 1e-12);
    EXPECT_EQ(bls.get_binned_weighted_square_values().size(), 10);
    // binned_weighted_square_values = sum(square(values - E(values))) / npts
    ASSERT_NEAR(Sum(bls.get_binned_weighted_square_values()), 24, 1e-12);
    EXPECT_EQ(bls.get_binned_weights().size(), 10);
    ASSERT_NEAR(Sum(bls.get_binned_weights()), 1, 1e-12);
    if (result.bls_result().power() > best_result.bls_result().power()) {
      best_result = result;
    }
  }
  // For period = 100, the box spans bins 3 = [30, 40) to 6 = [60, 70).
  EXPECT_EQ(best_result.nbins(), 10);
  EXPECT_EQ(best_result.bls_result().start(), 3);
  EXPECT_EQ(best_result.bls_result().width(), 4);
  EXPECT_FLOAT_EQ(best_result.bls_result().r(), 0.4);
  EXPECT_FLOAT_EQ(best_result.bls_result().s(), -2.4);
  EXPECT_FLOAT_EQ(best_result.bls_result().t(), 14.4);
  EXPECT_FLOAT_EQ(best_result.bls_result().total_signal(), 24);
  EXPECT_NEAR(best_result.bls_result().mse(), 0, 1e-12);
  EXPECT_FLOAT_EQ(best_result.period(), 100);
  EXPECT_FLOAT_EQ(best_result.duration(), 40);
  EXPECT_FLOAT_EQ(best_result.epoch(), 50);
  EXPECT_FLOAT_EQ(best_result.depth(), 10);
  EXPECT_FLOAT_EQ(best_result.baseline(), 20);
}

TEST(BoxLeastSquaresTest, FitNegativeMean) {
  // Create a time series evenly spaced at time intervals of width 10, starting
  // at 3, which periodically repeats the sequence {-5, -5, -5, 5, 5, 5, 5, -5,
  // -5, -5}. Its period is therefore 100.0.
  const int npts = 100;
  vector<double> time(npts);
  vector<double> values(npts);
  for (int i = 0; i < npts; ++i) {
    time[i] = i * 10 + 3;
    values[i] = (i % 10) >= 3 && (i % 10) < 7 ? 5 : -5;
  }

  // Initialize a BoxLeastSquares that can be used for different periods and
  // different numbers of bins. Set the initial capacity of the binning vectors
  // to 20.
  BoxLeastSquares bls(time, values, 20);
  EXPECT_FLOAT_EQ(bls.get_mean_value(), -1);

  // Consider periods between 10 and 200.
  vector<double> periods = {10,  20,  30,  40,  50,  60,  70,  80,  90,  100,
                            110, 120, 130, 140, 150, 160, 170, 180, 190, 200};

  // nbins = 10.
  // BLS options.
  int nbins = 10;
  BlsOptions options;

  // Output arguments.
  BoxTransitModel result;
  string error;

  // Run BLS on all periods.
  BoxTransitModel best_result;
  for (auto period : periods) {
    bls.Fit(period, nbins, options, &result, &error);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(bls.get_binned_weighted_values().size(), 10);
    ASSERT_NEAR(Sum(bls.get_binned_weighted_values()), 0, 1e-12);
    EXPECT_EQ(bls.get_binned_weighted_square_values().size(), 10);
    // binned_weighted_square_values = sum(square(values - E(values))) / npts
    ASSERT_NEAR(Sum(bls.get_binned_weighted_square_values()), 24, 1e-12);
    EXPECT_EQ(bls.get_binned_weights().size(), 10);
    ASSERT_NEAR(Sum(bls.get_binned_weights()), 1, 1e-12);
    if (result.bls_result().power() > best_result.bls_result().power()) {
      best_result = result;
    }
  }
  // For period = 100, the box spans bins 3 = [30, 40) to 6 = [60, 70).
  EXPECT_EQ(best_result.nbins(), 10);
  EXPECT_EQ(best_result.bls_result().start(), 3);
  EXPECT_EQ(best_result.bls_result().width(), 4);
  EXPECT_FLOAT_EQ(best_result.bls_result().r(), 0.4);
  EXPECT_FLOAT_EQ(best_result.bls_result().s(), 2.4);
  EXPECT_FLOAT_EQ(best_result.bls_result().t(), 14.4);
  EXPECT_FLOAT_EQ(best_result.bls_result().total_signal(), 24);
  EXPECT_NEAR(best_result.bls_result().mse(), 0, 1e-12);
  EXPECT_FLOAT_EQ(best_result.period(), 100);
  EXPECT_FLOAT_EQ(best_result.duration(), 40);
  EXPECT_FLOAT_EQ(best_result.epoch(), 50);
  EXPECT_FLOAT_EQ(best_result.depth(), -10);
  EXPECT_FLOAT_EQ(best_result.baseline(), -5);
}

}  // namespace
}  // namespace box_least_squares
}  // namespace exoplanet_ml
