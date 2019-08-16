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

// Box least squares algorithm for searching for a periodic box in a time
// series.
//
// E.g:
//
//   --------    --------    --------    --------    --------    --------
//           |  |        |  |        |  |        |  |        |  |
//            --          --          --          --          --

#ifndef TENSORFLOW_MODELS_ASTRONET_BOX_LEAST_SQUARES_BOX_LEAST_SQUARES_H_
#define TENSORFLOW_MODELS_ASTRONET_BOX_LEAST_SQUARES_BOX_LEAST_SQUARES_H_

#include <string>
#include <vector>

#include "box_least_squares/box_least_squares_impl.h"

namespace exoplanet_ml {
namespace box_least_squares {

// Runs the box least squares algorithm on an input series of values.
//
// This function finds the best-fitting box model for the input series assuming
// it is already phase folded on the desired period and binned if desired. To
// run box least squares on a non-phase-folded input series, and/or on multiple
// candidate periods, with binning, use the BoxLeastSquares class.
//
// Input args:
//   values: Values of input time series (e.g. flux).
//   weights: Weights of input values. Weights must be non-negative and be
//     proportional to the inverse variances of the values. Values with weight
//     zero are ignored.
//   options: A BlsOptions.
//
// Output args:
//   result: A BlsResult.
//   error: String indicating an error when running the algorithm (e.g. an
//     invalid argument).
//
// Returns:
//   true if the algorithm succeeded. If false, see "error".
bool RunBls(std::vector<double> values, std::vector<double> weights,
            BlsOptions options, BlsResult* result, string* error);

// Class for fitting box transit models using the box least squares algorithm.
//
// This class is mainly intended for use as a Python extension. It keeps the
// zero-centered light curve in the class state to minimize expensive copies
// between the language barrier when fitting multiple candidate periods.
//
// This class currently assumes that flux measurements all have the same
// variance.
class BoxLeastSquares {
 public:
  // Input args:
  //   time: Vector of time values of the time series.
  //   values: Vector of values of the time series.
  //   capacity: (Optional) Capacity of the internal vectors for binning. To
  //     avoid resizing the internal vectors, set this to the largest expected
  //     value of the `nbins` argument of Fit().
  explicit BoxLeastSquares(std::vector<double> time, std::vector<double> values,
                           const std::vector<double>::size_type capacity = 0);

  // Finds the best-fitting box model for a single candidate period.
  //
  // Input args:
  //   period: The candidate period. Must be positive.
  //   nbins: The number of bins for phase folding. Must be greater than 1.
  //   options: Options for the box least squares algorithm.
  //
  // Output args:
  //   error: String indicating an error when running the algorithm (e.g. an
  //     invalid argument).
  //
  // Returns:
  //   true if the algorithm succeeded. If false, see `error`.
  bool Fit(const double period, const int nbins, const BlsOptions& options,
           BoxTransitModel* result, string* error);

  // Getters.
  const std::vector<double>& get_time() const;
  const std::vector<double>& get_values() const;
  double get_mean_value() const;
  const std::vector<double>& get_binned_weighted_values() const;
  const std::vector<double>& get_binned_weighted_square_values() const;
  const std::vector<double>& get_binned_weights() const;

 private:
  // Input light curve.
  std::vector<double> time_;
  std::vector<double> values_;
  double mean_value_;

  // Binned time series used by box least squares.
  std::vector<double> binned_weighted_values_;
  std::vector<double> binned_weighted_square_values_;
  std::vector<double> binned_weights_;
};

}  // namespace box_least_squares
}  // namespace exoplanet_ml

#endif  // TENSORFLOW_MODELS_ASTRONET_BOX_LEAST_SQUARES_BOX_LEAST_SQUARES_H_
