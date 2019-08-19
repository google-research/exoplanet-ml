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

// Internal implementation of the box least squares algorithm, as described in:
//
//   "A box-fitting algorithm in the search for periodic transits."
//   G. Kovacs, S. Zucker and T. Mazeh, 2002.
//   Astronomy & Astrophysics, 391, 369.
//
// This algorithm solves a weighted least squares optimization problem to find
// the best-fitting partition of the input series into two levels:
//   1. The "inner level" == "box level"
//   2. The "outer level"
//
//    (Eg. 1):                                (Eg. 2):
//
//    ---------       -------------                 ------------------
//             |     |                             |                  |
//              -----                          ----                    ---
//             ^      ^                             ^                 ^
//             |      |                             |                 |
//           start   end                          start              end

#ifndef TENSORFLOW_MODELS_ASTRONET_BOX_LEAST_SQUARES_BOX_LEAST_SQUARES_IMPL_H_
#define TENSORFLOW_MODELS_ASTRONET_BOX_LEAST_SQUARES_BOX_LEAST_SQUARES_IMPL_H_

#include <string>
#include <vector>

#include "box_least_squares/box_least_squares.pb.h"

namespace exoplanet_ml {
namespace box_least_squares {
namespace internal {

// Runs the box least squares algorithm on an input series of weighted values.
//
// This internal implementation imposes preconditions on the inputs (e.g. that
// values are zero-centered and pre-multiplied by weights) to avoid making
// copies of the input vectors.
//
// Input args:
//   weighted_values: Weighted input series; that is,
//     weighted_values[i] = weights[i] * x[i], where x is the input series.
//     Must be zero-centered (i.e. sums to zero).
//   weighted_square_values: Weighted square values of the input series; that
//     is, weighted_square_values[i] = weights[i] * x[i]^2, where x is the input
//     series.
//   weights: Weights of each value. Each weight must be in [0, 1] and the sum
//     of all weights must be 1.
//   options: A BlsOptions.
//
// Output args:
//   result: A BlsResult.
//   error: String indicating an error when running the algorithm (e.g. an
//     invalid argument).
//
// Returns:
//   true if the algorithm succeeded. If false, see "error".
bool BlsImpl(const std::vector<double>& weighted_values,
             const std::vector<double>& weighted_square_values,
             const std::vector<double>& weights, BlsOptions options,
             BlsResult* result, std::string* error);

}  // namespace internal
}  // namespace box_least_squares
}  // namespace exoplanet_ml

#endif  // TENSORFLOW_MODELS_ASTRONET_BOX_LEAST_SQUARES_BOX_LEAST_SQUARES_IMPL_H_
