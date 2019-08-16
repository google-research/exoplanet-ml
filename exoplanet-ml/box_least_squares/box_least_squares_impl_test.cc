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

#include "box_least_squares/box_least_squares_impl.h"

#include <cmath>

#include "gtest/gtest.h"
#include "box_least_squares/box_least_squares.pb.h"
#include "box_least_squares/test_util.h"
#include "box_least_squares/util.h"

using std::vector;

namespace exoplanet_ml {
namespace box_least_squares {
namespace internal {
namespace {

class BoxLeastSquaresTest : public ::testing::Test {
 protected:
  // Output arguments.
  BlsResult result_;
  string error_;
};

TEST_F(BoxLeastSquaresTest, TooFewPoints) {
  vector<double> weighted_values = {0};
  vector<double> weighted_square_values = {0};
  vector<double> weights = {1};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);

  BlsOptions options;
  EXPECT_FALSE(BlsImpl(weighted_values, weighted_square_values, weights,
                       options, &result_, &error_));
  EXPECT_EQ(error_, "weighted_values must have at least 2 elements (got: 1)");
}

TEST_F(BoxLeastSquaresTest, UnequalVectorSizes1) {
  vector<double> weighted_values = {-1, 0, 1};
  vector<double> weighted_square_values = {2.5, 0, 2.5};
  vector<double> weights = {0.2, 0.2, 0.2, 0.2, 0.2};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);

  BlsOptions options;
  EXPECT_FALSE(BlsImpl(weighted_values, weighted_square_values, weights,
                       options, &result_, &error_));
  EXPECT_EQ(error_,
            "weighted_values.size() (got: 3) != weights.size() (got: 5)");
}

TEST_F(BoxLeastSquaresTest, UnequalVectorSizes2) {
  vector<double> weighted_values = {-1, 0, 1};
  vector<double> weighted_square_values = {2.5, 0, 2.5, 0};
  vector<double> weights = {0.4, 0.4, 0.2};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);

  BlsOptions options;
  EXPECT_FALSE(BlsImpl(weighted_values, weighted_square_values, weights,
                       options, &result_, &error_));
  EXPECT_EQ(error_,
            "weighted_values.size() (got: 3) != weighted_square_values.size() "
            "(got: 4)");
}

TEST_F(BoxLeastSquaresTest, WidthMinTooSmall) {
  // values = {-10, -5, 0, 5, 10}
  vector<double> weighted_values = {-2, -1, 0, 1, 2};
  vector<double> weighted_square_values = {20, 5, 0, 5, 20};
  vector<double> weights = {0.2, 0.2, 0.2, 0.2, 0.2};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);

  BlsOptions options;
  options.set_width_min(-1);
  EXPECT_FALSE(BlsImpl(weighted_values, weighted_square_values, weights,
                       options, &result_, &error_));
  EXPECT_EQ(error_, "width_min must be positive (got: -1)");
}

TEST_F(BoxLeastSquaresTest, WidthMaxTooLarge) {
  // values = {-10, -5, 0, 5, 10}
  vector<double> weighted_values = {-2, -1, 0, 1, 2};
  vector<double> weighted_square_values = {20, 5, 0, 5, 20};
  vector<double> weights = {0.2, 0.2, 0.2, 0.2, 0.2};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);

  BlsOptions options;
  options.set_width_max(5);
  EXPECT_FALSE(BlsImpl(weighted_values, weighted_square_values, weights,
                       options, &result_, &error_));
  EXPECT_EQ(error_, "width_max (got: 5) >= weighted_values.size (got: 5)");
}

TEST_F(BoxLeastSquaresTest, WidthMinGreaterThanWidthMax) {
  // values = {-10, -5, 0, 5, 10}
  vector<double> weighted_values = {-2, -1, 0, 1, 2};
  vector<double> weighted_square_values = {20, 5, 0, 5, 20};
  vector<double> weights = {0.2, 0.2, 0.2, 0.2, 0.2};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);

  BlsOptions options;
  options.set_width_min(3);
  options.set_width_max(2);
  EXPECT_FALSE(BlsImpl(weighted_values, weighted_square_values, weights,
                       options, &result_, &error_));
  EXPECT_EQ(error_, "width_min (got: 3) > width_max (got: 2)");
}

TEST_F(BoxLeastSquaresTest, WeightMinTooSmall) {
  // values = {-10, -5, 0, 5, 10}
  vector<double> weighted_values = {-2, -1, 0, 1, 2};
  vector<double> weighted_square_values = {20, 5, 0, 5, 20};
  vector<double> weights = {0.2, 0.2, 0.2, 0.2, 0.2};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);

  BlsOptions options;
  options.set_weight_min(-0.5);
  EXPECT_FALSE(BlsImpl(weighted_values, weighted_square_values, weights,
                       options, &result_, &error_));
  EXPECT_EQ(error_, "weight_min must be in [0, 1) (got: -0.5)");
}

TEST_F(BoxLeastSquaresTest, WeightMinTooLarge) {
  // values = {-10, -5, 0, 5, 10}
  vector<double> weighted_values = {-2, -1, 0, 1, 2};
  vector<double> weighted_square_values = {20, 5, 0, 5, 20};
  vector<double> weights = {0.2, 0.2, 0.2, 0.2, 0.2};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);

  BlsOptions options;
  options.set_weight_min(1);
  EXPECT_FALSE(BlsImpl(weighted_values, weighted_square_values, weights,
                       options, &result_, &error_));
  EXPECT_EQ(error_, "weight_min must be in [0, 1) (got: 1)");
}

TEST_F(BoxLeastSquaresTest, WeightMaxTooSmall) {
  // values = {-10, -5, 0, 5, 10}
  vector<double> weighted_values = {-2, -1, 0, 1, 2};
  vector<double> weighted_square_values = {20, 5, 0, 5, 20};
  vector<double> weights = {0.2, 0.2, 0.2, 0.2, 0.2};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);

  BlsOptions options;
  options.set_weight_max(-1);
  EXPECT_FALSE(BlsImpl(weighted_values, weighted_square_values, weights,
                       options, &result_, &error_));
  EXPECT_EQ(error_, "weight_max must be in (0, 1] (got: -1)");
}

TEST_F(BoxLeastSquaresTest, WeightMaxTooLarge) {
  // values = {-10, -5, 0, 5, 10}
  vector<double> weighted_values = {-2, -1, 0, 1, 2};
  vector<double> weights = {0.2, 0.2, 0.2, 0.2, 0.2};
  vector<double> weighted_square_values = {20, 5, 0, 5, 20};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);

  BlsOptions options;
  options.set_weight_max(1.5);
  EXPECT_FALSE(BlsImpl(weighted_values, weighted_square_values, weights,
                       options, &result_, &error_));
  EXPECT_EQ(error_, "weight_max must be in (0, 1] (got: 1.5)");
}

TEST_F(BoxLeastSquaresTest, PerfectFit1) {
  // values = {-30, 70, 70, 70, -30, -30, -30, -30, -30, -30}
  vector<double> weighted_values = {-3, 7, 7, 7, -3, -3, -3, -3, -3, -3};
  vector<double> weighted_square_values = {90, 490, 490, 490, 90,
                                           90, 90,  90,  90,  90};
  vector<double> weights = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);
  double total_signal = Sum(weighted_square_values);

  // Narrow box.
  BlsOptions options;
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
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
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
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

TEST_F(BoxLeastSquaresTest, PerfectFit2) {
  // values = {20, 20, 20, 20, 20, -30, -30, -30, -30, 20}
  vector<double> weighted_values = {2, 2, 2, 2, 2, -3, -3, -3, -3, 2};
  vector<double> weighted_square_values = {40, 40, 40, 40, 40,
                                           90, 90, 90, 90, 40};
  vector<double> weights = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);
  double total_signal = Sum(weighted_square_values);

  // Narrow box.
  BlsOptions options;
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
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
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
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

TEST_F(BoxLeastSquaresTest, PerfectFit3) {
  // values = {70, 70, -30, -30, -30, -30, -30, -30, -30, 70}
  vector<double> weighted_values = {7, 7, -3, -3, -3, -3, -3, -3, -3, 7};
  vector<double> weighted_square_values = {490, 490, 90, 90, 90,
                                           90,  90,  90, 90, 490};
  vector<double> weights = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);
  double total_signal = Sum(weighted_square_values);

  // Narrow box.
  BlsOptions options;
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 9);
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
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 2);
  EXPECT_EQ(result_.width(), 7);
  EXPECT_FLOAT_EQ(result_.r(), 0.7);
  EXPECT_FLOAT_EQ(result_.s(), -21);
  EXPECT_FLOAT_EQ(result_.t(), 630);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_FLOAT_EQ(result_.power(), total_signal);  // Perfect fit.
  EXPECT_FLOAT_EQ(result_.mse(), 0);               // Perfect fit.
}

TEST_F(BoxLeastSquaresTest, ImperfectFit) {
  // values = {21, 20, 19, 20, 21, -32, -30, -28, -30, 19}
  vector<double> weighted_values = {2.1,  2,  1.9,  2,  2.1,
                                    -3.2, -3, -2.8, -3, 1.9};
  vector<double> weighted_square_values = {44.1,  40.0, 36.1, 40.0, 44.1,
                                           102.4, 90.0, 78.4, 90.0, 36.1};
  vector<double> weights = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);
  double total_signal = Sum(weighted_square_values);

  // Narrow box.
  BlsOptions options;
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 5);
  EXPECT_EQ(result_.width(), 4);
  EXPECT_FLOAT_EQ(result_.r(), 0.4);
  EXPECT_FLOAT_EQ(result_.s(), -12);
  EXPECT_FLOAT_EQ(result_.t(), 360.8);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_LT(result_.power(), total_signal);  // Imperfect fit.
  EXPECT_GT(result_.mse(), 0);               // Imperfect fit.

  // Wide box.
  options.set_width_min(5);
  options.set_width_max(9);
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 9);
  EXPECT_EQ(result_.width(), 6);
  EXPECT_FLOAT_EQ(result_.r(), 0.6);
  EXPECT_FLOAT_EQ(result_.s(), 12);
  EXPECT_FLOAT_EQ(result_.t(), 240.4);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_LT(result_.power(), total_signal);  // Imperfect fit.
  EXPECT_GT(result_.mse(), 0);               // Imperfect fit.
}

TEST_F(BoxLeastSquaresTest, SingleBinBox) {
  // values = {10, -90, 10, 10, 10, 10, 10, 10, 10, 10}
  vector<double> weighted_values = {1, -9, 1, 1, 1, 1, 1, 1, 1, 1};
  vector<double> weighted_square_values = {10, 810, 10, 10, 10,
                                           10, 10,  10, 10, 10};
  vector<double> weights = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);
  double total_signal = Sum(weighted_square_values);

  // Narrow box.
  BlsOptions options;
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 1);
  EXPECT_EQ(result_.width(), 1);
  EXPECT_FLOAT_EQ(result_.r(), 0.1);
  EXPECT_FLOAT_EQ(result_.s(), -9);
  EXPECT_FLOAT_EQ(result_.t(), 810);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_FLOAT_EQ(result_.power(), total_signal);  // Perfect fit.
  EXPECT_FLOAT_EQ(result_.mse(), 0);               // Perfect fit.

  // Wide box.
  options.set_width_min(2);
  options.set_width_max(9);
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 2);
  EXPECT_EQ(result_.width(), 9);
  EXPECT_FLOAT_EQ(result_.r(), 0.9);
  EXPECT_FLOAT_EQ(result_.s(), 9);
  EXPECT_FLOAT_EQ(result_.t(), 90);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_FLOAT_EQ(result_.power(), total_signal);  // Perfect fit.
  EXPECT_FLOAT_EQ(result_.mse(), 0);               // Perfect fit.
}

TEST_F(BoxLeastSquaresTest, ZeroWeightsOffBoxBoundary) {
  // values = {0, 20, 20, 20, 0, 20, 20, -30, 0, -30, -30, -30, 20, 0}
  vector<double> weighted_values = {0,  2, 2,  2,  0,  2, 2,
                                    -3, 0, -3, -3, -3, 2, 0};
  vector<double> weighted_square_values = {0,  40, 40, 40, 0,  40, 40,
                                           90, 0,  90, 90, 90, 40, 0};
  vector<double> weights = {0,   0.1, 0.1, 0.1, 0,   0.1, 0.1,
                            0.1, 0,   0.1, 0.1, 0.1, 0.1, 0};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);
  double total_signal = Sum(weighted_square_values);

  // Narrow box.
  BlsOptions options;
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 7);
  EXPECT_EQ(result_.width(), 5);
  EXPECT_FLOAT_EQ(result_.r(), 0.4);
  EXPECT_FLOAT_EQ(result_.s(), -12);
  EXPECT_FLOAT_EQ(result_.t(), 360);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_FLOAT_EQ(result_.power(), total_signal);  // Perfect fit.
  EXPECT_FLOAT_EQ(result_.mse(), 0);               // Perfect fit.

  // Wide box.
  options.set_width_min(6);
  options.set_width_max(13);
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 12);
  EXPECT_EQ(result_.width(), 9);
  EXPECT_FLOAT_EQ(result_.r(), 0.6);
  EXPECT_FLOAT_EQ(result_.s(), 12);
  EXPECT_FLOAT_EQ(result_.t(), 240);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_FLOAT_EQ(result_.power(), total_signal);  // Perfect fit.
  EXPECT_FLOAT_EQ(result_.mse(), 0);               // Perfect fit.
}

TEST_F(BoxLeastSquaresTest, ZeroWeightsOnBoxBoundary) {
  // The box should skip zero-weight points on the boundary, even though they
  // would not change the power.
  // values = {20, 20, 20, 0, -30, -30, -30, -30, 0, 20, 20, 20}
  vector<double> weighted_values = {2, 2, 2, 0, -3, -3, -3, -3, 0, 2, 2, 2};
  vector<double> weighted_square_values = {40, 40, 40, 0,  90, 90,
                                           90, 90, 0,  40, 40, 40};
  vector<double> weights = {0.1, 0.1, 0.1, 0,   0.1, 0.1,
                            0.1, 0.1, 0,   0.1, 0.1, 0.1};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);
  double total_signal = Sum(weighted_square_values);

  // Narrow box.
  BlsOptions options;
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 4);
  EXPECT_EQ(result_.width(), 4);
  EXPECT_FLOAT_EQ(result_.r(), 0.4);
  EXPECT_FLOAT_EQ(result_.s(), -12);
  EXPECT_FLOAT_EQ(result_.t(), 360);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_FLOAT_EQ(result_.power(), total_signal);  // Perfect fit.
  EXPECT_FLOAT_EQ(result_.mse(), 0);               // Perfect fit.

  // Wide box.
  options.set_width_min(5);
  options.set_width_max(11);
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
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

TEST_F(BoxLeastSquaresTest, LowWeightOutlier) {
  // values = {18, 18, 18, 18, -36, -36, 0, -36, 18, 18}
  vector<double> weighted_values = {2, 2, 2, 2, -4, -4, 0, -4, 2, 2};
  vector<double> weighted_square_values = {36,  36, 36,  36, 144,
                                           144, 0,  144, 36, 36};
  vector<double> weights = {0.11, 0.11, 0.11, 0.11, 0.11,
                            0.11, 0.01, 0.11, 0.11, 0.11};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);
  double total_signal = Sum(weighted_square_values);

  // There are two equivalent best-fit boxes. Check that it came up with one.
  BlsOptions options;
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 4);
  EXPECT_EQ(result_.width(), 4);
  EXPECT_FLOAT_EQ(result_.r(), 0.34);
  EXPECT_FLOAT_EQ(result_.s(), -12);
  EXPECT_FLOAT_EQ(result_.t(), 432);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_LT(result_.power(), total_signal);  // Imperfect fit.
  EXPECT_GT(result_.mse(), 0);               // Imperfect fit.
}

TEST_F(BoxLeastSquaresTest, HalfHalfInput) {
  // values = {-10, -10, -10, -10, -10, 10, 10, 10, 10, 10}
  vector<double> weighted_values = {-1, -1, -1, -1, -1, 1, 1, 1, 1, 1};
  vector<double> weighted_square_values = {10, 10, 10, 10, 10,
                                           10, 10, 10, 10, 10};
  vector<double> weights = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);
  double total_signal = Sum(weighted_square_values);

  // There are two equivalent perfect-fit boxes. Check that it came up with
  // one.
  BlsOptions options;
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_TRUE(result_.start() == 0 || result_.start() == 5);
  EXPECT_EQ(result_.width(), 5);
  EXPECT_FLOAT_EQ(result_.r(), 0.5);
  EXPECT_FLOAT_EQ(std::abs(result_.s()), 5);
  EXPECT_FLOAT_EQ(result_.t(), 50);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_FLOAT_EQ(result_.power(), total_signal);  // Perfect fit.
  EXPECT_FLOAT_EQ(result_.mse(), 0);               // Perfect fit.
}

TEST_F(BoxLeastSquaresTest, LeftAlignedBox) {
  // values = {-30, -30, -30, -30, 20, 20, 20, 20, 20, 20}
  vector<double> weighted_values = {-3, -3, -3, -3, 2, 2, 2, 2, 2, 2};
  vector<double> weighted_square_values = {90, 90, 90, 90, 40,
                                           40, 40, 40, 40, 40};
  vector<double> weights = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);
  double total_signal = Sum(weighted_square_values);

  // Narrow box.
  BlsOptions options;
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 0);
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
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 4);
  EXPECT_EQ(result_.width(), 6);
  EXPECT_FLOAT_EQ(result_.r(), 0.6);
  EXPECT_FLOAT_EQ(result_.s(), 12);
  EXPECT_FLOAT_EQ(result_.t(), 240);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_FLOAT_EQ(result_.power(), total_signal);  // Perfect fit.
  EXPECT_FLOAT_EQ(result_.mse(), 0);               // Perfect fit.
}

TEST_F(BoxLeastSquaresTest, RightAlignedBox) {
  // values = {-30, -30, -30, -30, 20, 20, 20, 20, 20, 20}
  vector<double> weighted_values = {-3, -3, -3, -3, 2, 2, 2, 2, 2, 2};
  vector<double> weighted_square_values = {90, 90, 90, 90, 40,
                                           40, 40, 40, 40, 40};
  vector<double> weights = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);
  double total_signal = Sum(weighted_square_values);

  // Narrow box.
  BlsOptions options;
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 0);
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
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 4);
  EXPECT_EQ(result_.width(), 6);
  EXPECT_FLOAT_EQ(result_.r(), 0.6);
  EXPECT_FLOAT_EQ(result_.s(), 12);
  EXPECT_FLOAT_EQ(result_.t(), 240);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_FLOAT_EQ(result_.power(), total_signal);  // Perfect fit.
  EXPECT_FLOAT_EQ(result_.mse(), 0);               // Perfect fit.
}

TEST_F(BoxLeastSquaresTest, BoxForcedTooBig) {
  // values = {20, 20, 20, 20, 18, -30, -30, -30, -28, 20}
  vector<double> weighted_values = {2, 2, 2, 2, 1.9, -3, -3, -3, -2.9, 2};
  vector<double> weighted_square_values = {40.0, 40.0, 40.0, 40.0, 34.2,
                                           90.0, 90.0, 90.0, 81.2, 40.0};
  vector<double> weights = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);
  double total_signal = Sum(weighted_square_values);

  // There are two equivalent best-fit boxes. Check that it came up with one.
  BlsOptions options;
  options.set_width_min(5);
  options.set_width_max(5);
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_TRUE(result_.start() == 4 || result_.start() == 9);
  EXPECT_EQ(result_.width(), 5);
  EXPECT_FLOAT_EQ(result_.r(), 0.5);
  EXPECT_FLOAT_EQ(std::abs(result_.s()), 10);
  if (result_.start() == 4) {
    EXPECT_FLOAT_EQ(std::abs(result_.t()), 385.4);
  } else {
    EXPECT_FLOAT_EQ(std::abs(result_.t()), 200);
  }
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_LT(result_.power(), total_signal);  // Imperfect fit.
  EXPECT_GT(result_.mse(), 0);               // Imperfect fit.
}

TEST_F(BoxLeastSquaresTest, BoxForcedTooSmall) {
  // values = {18, 20, 20, 20, 20, -30, -30, -30, -28, 20}
  vector<double> weighted_values = {1.9, 2, 2, 2, 2, -3, -3, -3, -2.9, 2};
  vector<double> weighted_square_values = {34.2, 40.0, 40.0, 40.0, 40.0,
                                           90.0, 90.0, 90.0, 81.2, 40.0};
  vector<double> weights = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);
  double total_signal = Sum(weighted_square_values);

  // Narrow box.
  BlsOptions options;
  options.set_width_max(3);
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 5);
  EXPECT_EQ(result_.width(), 3);
  EXPECT_FLOAT_EQ(result_.r(), 0.3);
  EXPECT_FLOAT_EQ(result_.s(), -9);
  EXPECT_FLOAT_EQ(result_.t(), 270);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_LT(result_.power(), total_signal);  // Imperfect fit.
  EXPECT_GT(result_.mse(), 0);               // Imperfect fit.
}

TEST_F(BoxLeastSquaresTest, BoxLimitedByWeight) {
  // values = {1.0, 1.0, 1.0, 1.0, 1.1, -99.0, 0.9, 1.0, 1.0, 1.0}
  vector<double> weighted_values = {0.11,  0.11,  0.11, 0.11, 0.121,
                                    -0.99, 0.099, 0.11, 0.11, 0.11};
  vector<double> weighted_square_values = {0.11,  0.11,   0.11, 0.11, 0.1331,
                                           98.01, 0.0891, 0.11, 0.11, 0.11};
  vector<double> weights = {0.11, 0.11, 0.11, 0.11, 0.11,
                            0.01, 0.11, 0.11, 0.11, 0.11};
  ASSERT_NEAR(Sum(weights), 1, 1e-12);
  ASSERT_NEAR(Sum(weighted_values), 0, 1e-12);
  double total_signal = Sum(weighted_square_values);

  // With no weight_min, the best box has one element.
  BlsOptions options;
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 5);
  EXPECT_EQ(result_.width(), 1);
  EXPECT_FLOAT_EQ(result_.r(), 0.01);
  EXPECT_FLOAT_EQ(result_.s(), -0.99);
  EXPECT_FLOAT_EQ(result_.t(), 98.01);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_LT(result_.power(), total_signal);  // Imperfect fit.
  EXPECT_GT(result_.mse(), 0);               // Imperfect fit.

  // weight_min prevents the single-element best fit box.
  options.set_weight_min(0.02);
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 5);
  EXPECT_EQ(result_.width(), 2);
  EXPECT_FLOAT_EQ(result_.r(), 0.12);
  EXPECT_FLOAT_EQ(result_.s(), -0.891);
  EXPECT_FLOAT_EQ(result_.t(), 98.0991);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_LT(result_.power(), total_signal);  // Imperfect fit.
  EXPECT_GT(result_.mse(), 0);               // Imperfect fit.

  // A wide box can use the single-element best fit box, because there's no
  // weight_max.
  options.set_width_min(3);
  options.set_width_max(9);
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 6);
  EXPECT_EQ(result_.width(), 9);
  EXPECT_FLOAT_EQ(result_.r(), 0.99);
  EXPECT_FLOAT_EQ(result_.s(), 0.99);
  EXPECT_FLOAT_EQ(result_.t(), 0.9922);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_LT(result_.power(), total_signal);  // Imperfect fit.
  EXPECT_GT(result_.mse(), 0);               // Imperfect fit.

  // weight_max prevents the single-element best fit box.
  options.set_weight_max(0.98);
  ASSERT_TRUE(BlsImpl(weighted_values, weighted_square_values, weights, options,
                      &result_, &error_));
  EXPECT_TRUE(error_.empty());
  EXPECT_EQ(result_.start(), 7);
  EXPECT_EQ(result_.width(), 8);
  EXPECT_FLOAT_EQ(result_.r(), 0.88);
  EXPECT_FLOAT_EQ(result_.s(), 0.891);
  EXPECT_FLOAT_EQ(result_.t(), 0.9031);
  EXPECT_FLOAT_EQ(result_.total_signal(), total_signal);
  EXPECT_LT(result_.power(), total_signal);  // Imperfect fit.
  EXPECT_GT(result_.mse(), 0);               // Imperfect fit.
}

}  // namespace
}  // namespace internal
}  // namespace box_least_squares
}  // namespace exoplanet_ml
