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

#ifndef EXOPLANET_ML_BOX_LEAST_SQUARES_UTIL_H_
#define EXOPLANET_ML_BOX_LEAST_SQUARES_UTIL_H_

#include <numeric>
#include <vector>

namespace exoplanet_ml {
namespace box_least_squares {

// Returns the sum of elements of a vector.
inline double Sum(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0);
}

}  // namespace box_least_squares
}  // namespace exoplanet_ml

#endif  // EXOPLANET_ML_BOX_LEAST_SQUARES_UTIL_H_
