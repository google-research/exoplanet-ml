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

#include "absl/strings/substitute.h"
#include "box_least_squares/box_least_squares.pb.h"

using absl::Substitute;
using std::vector;

namespace exoplanet_ml {
namespace box_least_squares {
namespace internal {

bool ValidateBlsInput(const vector<double>& weighted_values,
                      const vector<double>& weighted_square_values,
                      const vector<double>& weights, BlsOptions* options,
                      string* error) {
  // Validate time series length.
  if (weighted_values.size() < 2) {
    *error =
        Substitute("weighted_values must have at least 2 elements (got: $0)",
                   weighted_values.size());
    return false;
  }
  if (weighted_values.size() != weighted_square_values.size()) {
    *error = Substitute(
        "weighted_values.size() (got: $0) != weighted_square_values.size() "
        "(got: $1)",
        weighted_values.size(), weighted_square_values.size());
    return false;
  }
  if (weighted_values.size() != weights.size()) {
    *error = Substitute(
        "weighted_values.size() (got: $0) != weights.size() (got: $1)",
        weighted_values.size(), weights.size());
    return false;
  }

  // Set default options. Note that an unset field has value 0 in proto3.
  if (options->width_min() == 0) {
    options->set_width_min(1);
  }
  if (options->width_max() == 0) {
    options->set_width_max(weighted_values.size() / 2);
  }
  if (options->weight_max() == 0) {
    options->set_weight_max(1);
  }

  // Validate width_{min,max}.
  if (options->width_min() <= 0) {
    *error = Substitute("width_min must be positive (got: $0)",
                        options->width_min());
    return false;
  }
  if (options->width_max() >= weighted_values.size()) {
    *error = Substitute("width_max (got: $0) >= weighted_values.size (got: $1)",
                        options->width_max(), weighted_values.size());
    return false;
  }
  if (options->width_min() > options->width_max()) {
    *error = Substitute("width_min (got: $0) > width_max (got: $1)",
                        options->width_min(), options->width_max());
    return false;
  }

  // Validate weight_{min,max}.
  if (options->weight_min() < 0 || options->weight_min() >= 1) {
    *error = Substitute("weight_min must be in [0, 1) (got: $0)",
                        options->weight_min());
    return false;
  }
  if (options->weight_max() <= 0 || options->weight_max() > 1) {
    *error = Substitute("weight_max must be in (0, 1] (got: $0)",
                        options->weight_max());
    return false;
  }
  return true;
}

bool BlsImpl(const vector<double>& weighted_values,
             const vector<double>& weighted_square_values,
             const vector<double>& weights, BlsOptions options,
             BlsResult* result, string* error) {
  // Validate input.
  if (!ValidateBlsInput(weighted_values, weighted_square_values, weights,
                        &options, error)) {
    return false;
  }
  result->Clear();

  // Define the search indices. At each step, we consider the box in the index
  // interval [start, end]. Note that both indices are inclusive. We allow the
  // end index to "wrap", so it is possible that start > end, in which case the
  // box consists of indices [start, npts - 1] U [0, end]. The width parameter
  // indicates the number of points in the current box.
  int start;
  int end;
  int width;

  // Define the box least squares parameters.
  // r is the sum of weights for all indices in [start, end].
  // s is the sum of weighted_values for all indices in [start, end].
  // t is the sum of weighted_square_values for all indices in [start, end].
  // power is the BLS maximization score, equal to s^2 / (r * (1 - r)).
  // total_signal is the sum of weighted_square_values. The mean squared
  //   deviation of the data from the model is (total_signal - power).
  double r;
  double s;
  double t;
  double power;
  double total_signal = 0;

  // For a given value of start, {r,s,t}_start are the respective values of r,
  // s, and t for the box interval [start, start + width_min - 1). Note that
  // this interval is one point narrower than the minimum box width, because we
  // always increment it by one point before computing the BLS power. Note also
  // that this interval might be empty (if width_min = 1), in which case
  // r_start = s_start = 0.
  double r_start = 0;
  double s_start = 0;
  double t_start = 0;
  for (int i = 0; i < options.width_min() - 1; ++i) {
    r_start += weights[i];
    s_start += weighted_values[i];
    t_start += weighted_square_values[i];
  }

  // Search for boxes at every start index.
  const vector<double>::size_type npts = weighted_values.size();
  for (start = 0; start < npts; ++start) {
    total_signal += weighted_square_values[start];
    if (weights[start] > 0) {
      // Initialize r, s, t for the box interval [start, start + width_min - 1).
      r = r_start;
      s = s_start;
      t = t_start;

      // Search for boxes with width between width_min and width_max.
      for (width = options.width_min(); width <= options.width_max(); ++width) {
        // Update the end index, which might "wrap".
        end = (start + width - 1) % npts;

        if (weights[end] > 0) {
          // Update r and s for the new end index.
          r += weights[end];
          s += weighted_values[end];
          t += weighted_square_values[end];

          // Update result if this is the best fitting box so far.
          if (r > 0 && r < 1 && r >= options.weight_min() &&
              r <= options.weight_max()) {
            power = s * s / (r * (1 - r));  // r * (1 - r) is always in (0, 1).
            if (power > result->power()) {
              result->set_start(start);
              result->set_width(width);
              result->set_r(r);
              result->set_s(s);
              result->set_t(t);
              result->set_power(power);
            }
          }
        }
      }
    }

    // Prepare for the next start index.
    if (options.width_min() > 1) {
      end = (start + options.width_min() - 1) % npts;
      r_start += (weights[end] - weights[start]);
      s_start += (weighted_values[end] - weighted_values[start]);
      t_start += (weighted_square_values[end] - weighted_square_values[start]);
    }
  }

  // Compute the mean squared deviation of the data from the best fitting box.
  result->set_total_signal(total_signal);
  result->set_mse(total_signal - result->power());

  return true;
}

}  // namespace internal
}  // namespace box_least_squares
}  // namespace exoplanet_ml
