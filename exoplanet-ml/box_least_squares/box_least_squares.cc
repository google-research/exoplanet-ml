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

#include <cmath>
#include <numeric>
#include <utility>

#include "absl/strings/substitute.h"
#include "box_least_squares/bin_by_phase.h"
#include "box_least_squares/box_least_squares_impl.h"
#include "box_least_squares/util.h"

using absl::Substitute;
using std::vector;

namespace exoplanet_ml {
namespace box_least_squares {

bool RunBls(vector<double> values, vector<double> weights, BlsOptions options,
            BlsResult* result, std::string* error) {
  // Validate input lengths.
  if (values.size() != weights.size()) {
    *error = Substitute("values.size() (got: $0) != weights.size() (got: $1)",
                        values.size(), weights.size());
    return false;
  }

  // Zero center the values.
  double mean_value = Sum(values) / values.size();
  for (double& value : values) value -= mean_value;

  // Normalize the weights.
  double total_weight = Sum(weights);
  if (total_weight <= 0) {
    *error =
        Substitute("Total weight must be positive (got: $0)", total_weight);
    return false;
  }
  for (double& weight : weights) weight /= total_weight;

  // Compute weighted values and weighted square values.
  vector<double> weighted_values(values.size());
  vector<double> weighted_square_values(values.size());
  for (int i = 0; i < values.size(); ++i) {
    if (weights[i] < 0) {
      *error = Substitute("weights must be nonnegative (got: $0 at index $1)",
                          weights[i], i);
      return false;
    }
    weighted_values[i] = weights[i] * values[i];
    weighted_square_values[i] = weighted_values[i] * values[i];
  }

  // Run BLS.
  return internal::BlsImpl(weighted_values, weighted_square_values, weights,
                           options, result, error);
}

BoxLeastSquares::BoxLeastSquares(vector<double> time, vector<double> values,
                                 const std::vector<double>::size_type capacity)
    : time_(std::move(time)),
      values_(std::move(values)),
      mean_value_(0.0),
      binned_weighted_values_(capacity),
      binned_weighted_square_values_(capacity),
      binned_weights_(capacity) {
  // Zero-center the values.
  mean_value_ =
      std::accumulate(values_.begin(), values_.end(), 0.0) / values_.size();
  for (double& value : values_) value -= mean_value_;
}

bool BoxLeastSquares::Fit(const double period, const int nbins,
                          const BlsOptions& options, BoxTransitModel* result,
                          std::string* error) {
  // Set BLS options.
  result->set_nbins(nbins);
  *result->mutable_options() = options;

  // Bin by phase.
  if (!BinByPhase(time_, values_, period, nbins, &binned_weighted_values_,
                  &binned_weighted_square_values_, &binned_weights_, error)) {
    return false;
  }

  // For each bin index i, let us denote:
  //   bin_counts[i] = the number of points in each bin.
  //   bin_sums[i] = the sum of points in each bin.
  //   bin_square_sums[i] = the sum of squares of points in each bin.
  //
  // After BinByPhase(), we currently have:
  //   binned_weights_[i] = bin_counts[i]
  //   binned_weighted_values_[i] = bin_sums[i]
  //   binned_weighted_square_values_[i] = bin_square_sums[i]
  //
  // To prepare for BlsImpl(), we want:
  //   binned_weights_[i] = bin_counts[i] / npoints
  //   binned_weighted_values_[i] = bin_sums[i] / npoints
  //   binned_weighted_square_values_[i] = bin_square_sums[i] / npoints
  //
  // Thus, we divide each element of binned_weights_, binned_weighted_values_,
  // and binned_weighted_square_values_ by npoints.
  auto npoints = time_.size();
  for (int i = 0; i < binned_weights_.size(); ++i) {
    binned_weights_[i] /= npoints;
    binned_weighted_values_[i] /= npoints;
    binned_weighted_square_values_[i] /= npoints;
  }

  // Run Box Least Squares.
  BlsResult* bls_result = result->mutable_bls_result();
  if (!internal::BlsImpl(binned_weighted_values_,
                         binned_weighted_square_values_, binned_weights_,
                         options, bls_result, error)) {
    return false;
  }

  // Compute transit parameters.
  double bin_width = period / nbins;
  double transit_start = bls_result->start() * bin_width;
  double duration = bls_result->width() * bin_width;
  double depth = -bls_result->power() / bls_result->s();

  result->set_period(period);
  result->set_duration(duration);
  result->set_epoch(fmod(transit_start + duration / 2, period));
  result->set_depth(depth);
  result->set_baseline(bls_result->s() / bls_result->r() + depth + mean_value_);

  return true;
}

const vector<double>& BoxLeastSquares::get_time() const { return time_; }
const vector<double>& BoxLeastSquares::get_values() const { return values_; }
double BoxLeastSquares::get_mean_value() const { return mean_value_; }
const vector<double>& BoxLeastSquares::get_binned_weighted_values() const {
  return binned_weighted_values_;
}
const vector<double>& BoxLeastSquares::get_binned_weighted_square_values()
    const {
  return binned_weighted_square_values_;
}
const vector<double>& BoxLeastSquares::get_binned_weights() const {
  return binned_weights_;
}

}  // namespace box_least_squares
}  // namespace exoplanet_ml
