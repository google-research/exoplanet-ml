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

#ifndef TENSORFLOW_MODELS_ASTRONET_BOX_LEAST_SQUARES_BIN_BY_PHASE_H_
#define TENSORFLOW_MODELS_ASTRONET_BOX_LEAST_SQUARES_BIN_BY_PHASE_H_

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"

namespace exoplanet_ml {
namespace box_least_squares {

namespace internal {

template <typename RawType>
void ResizeAndClear(const typename std::vector<RawType>::size_type size,
                    std::vector<RawType>* v) {
  v->resize(0);
  v->resize(size);  // Fills with default-constructed items (zero for numbers).
}

}  // namespace internal

// Bins a time series by phase, relative to a specified period.
//
// The input time series consists of pairs of points (time, value). The values
// are accumulated based on their phase relative to the specified period.
//
// Bin i represents the following half-open interval on the time axis:
//     [i * period / nbins, (i + 1) * period / nbins).
//
// Conversely, the bin index for a particular time point is given by:
//     i = floor((time mod period) / (period / nbins)).
//
// Note that the bin of a time point t is undefined if the bin width
// (period / nbins) exactly divides (t mod period). In that case, the bin index
// should be i, where i = (t mod period) / (period / nbins), but due to floating
// point arithmetic, the right hand side may be slightly less than i, which
// would then be floored to i-1.
//
// Input args:
//   time: Vector of time values of the time series. Values must be nonnegative.
//   values: Vector of values of the time series.
//   period: A candidate period. Must be positive.
//
// Output args:
//   binned_values: The sum of values in each bin after phase folding.
//   binned_square_values: The sum of squares of values in each bin after phase
//     folding.
//   bin_counts: The number of points in each bin.
//   error: String indicating an error (e.g. an invalid argument).
//
// Returns:
//   true if the binning succeeded. If false, see "error".
template <typename TimeType, typename ValueType, typename CountType>
bool BinByPhase(const std::vector<TimeType>& time,
                const std::vector<ValueType>& values, const double period,
                const int nbins, std::vector<ValueType>* binned_values,
                std::vector<ValueType>* binned_square_values,
                std::vector<CountType>* bin_counts, string* error) {
  if (time.empty()) {
    *error = "time must not be empty";
    return false;
  }
  if (time.size() != values.size()) {
    *error =
        absl::Substitute("time.size() (got: $0) != values.size() (got: $1)",
                         time.size(), values.size());
    return false;
  }
  if (period <= 0) {
    *error = absl::Substitute("period must be positive (got: $0)", period);
    return false;
  }
  if (nbins <= 0) {
    *error = absl::Substitute("nbins must be positive (got: $0)", nbins);
    return false;
  }

  // Clear binned_values and bin_counts.
  internal::ResizeAndClear(nbins, binned_values);
  internal::ResizeAndClear(nbins, binned_square_values);
  internal::ResizeAndClear(nbins, bin_counts);

  // Phase fold.
  const double bin_width_inv = nbins / period;  // Reciprocal of bin width.
  int bin_index;
  double value;
  for (int i = 0; i < time.size(); ++i) {
    bin_index = static_cast<int>(bin_width_inv * fmod(time[i], period));
    value = values[i];
    (*binned_values)[bin_index] += value;
    (*binned_square_values)[bin_index] += (value * value);
    ++(*bin_counts)[bin_index];
  }
  return true;
}

}  // namespace box_least_squares
}  // namespace exoplanet_ml

#endif  // TENSORFLOW_MODELS_ASTRONET_BOX_LEAST_SQUARES_BIN_BY_PHASE_H_
