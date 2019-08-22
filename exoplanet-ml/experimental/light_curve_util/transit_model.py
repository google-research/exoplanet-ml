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

"""Functions for fitting and evaluating transit models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import batman
import numpy as np
import scipy.optimize

DEFAULT_EXPOSURE_TIME = 1764.944 / (24 * 3600)  # For Kepler.


def compute_semi_major_axis(period, duration, planet_radius, impact):
  """Computes the semi major axis in units of stellar radii.

  This computation assumes a circular orbit (i.e. the semi-major axis is the
  radius of the circular orbit).

  See Eq. (8) in https://arxiv.org/pdf/astro-ph/0206228.pdf.

  Args:
    period: Orbital period, in the same units as transit duration.
    duration: Total transit duration (first to fourth contact), in the same
      units as period.
    planet_radius: Planet radius, in units of stellar radius.
    impact: Impact parameter.

  Returns:
    The semi major axis in units of stellar radii.

  Raises:
    ValueError: If any of the parameters are nonpositive.
  """
  for name, value in [("Period", period), ("Duration", duration),
                      ("Planet radius", planet_radius)]:
    if value <= 0:
      raise ValueError("{} must be positive. Got: {}".format(name, value))

  if impact < 0 or impact > 1:
    raise ValueError("Impact must be in [0, 1]. Got: {}".format(impact))

  x = np.sin(duration * np.pi / period)**2
  return np.sqrt(((1 + planet_radius)**2 - impact**2 * (1 - x)) / x)


def compute_duration(period, planet_radius, semi_major_axis, inclination):
  """Computes the total transit duration (first to fourth contact).

  This computation assumes a circular orbit.

  See Eq. (3) in https://arxiv.org/pdf/astro-ph/0206228.pdf.

  Args:
    period: Orbital period.
    planet_radius: Planet radius, in units of stellar radius.
    semi_major_axis: Radius of the circular orbit, in units of stellar radius.
    inclination: Orbital inclination, in degrees.

  Returns:
    The total transit duration (first to fourth contact), in the same units as
    period.

  Raises:
    ValueError: If any of the parameters are nonpositive.
  """
  for name, value in [("Period", period), ("Planet radius", planet_radius),
                      ("Semi major axis", semi_major_axis),
                      ("Inclination", inclination)]:
    if value <= 0:
      raise ValueError("{} must be positive. Got: {}".format(name, value))

  cos_i = np.cos(inclination * np.pi / 180)
  x = ((1 + planet_radius)**2 - (semi_major_axis * cos_i)**2) / (1 - cos_i**2)
  return (period / np.pi) * np.arcsin(np.sqrt(x) / semi_major_axis)


def compute_inclination(impact, semi_major_axis):
  """Computes the orbital inclination, in degrees.

  This computation assumes a circular orbit.

  See Eq. (7) in https://arxiv.org/pdf/astro-ph/0206228.pdf.

  Args:
    impact: Impact parameter.
    semi_major_axis: Radius of the circular orbit, in units of stellar radius.

  Returns:
    The orbital inclination, in degrees.
  """
  return np.arccos(impact / semi_major_axis) * 180 / np.pi


def transit_params(t0,
                   period,
                   duration,
                   planet_radius,
                   impact,
                   limb_dark_model="uniform",
                   limb_dark_coeff=None):
  """Helper function to create a batman.TransitParams assuming a circular orbit.

  This function re-parametrizes the batman transit model by replacing the
  semi-major axis and orbital inclination parameters with transit duration and
  impact parameters. The latter parameters are preferred when fitting a model
  to an observed transit in a light curve, because:
    1. Transit duration can be directly estimated from the light curve.
    2. Impact parameter can be constrained to lie within [0, 1], whereas the
       equivalent constraint in terms of orbital inclination would depend on the
       semi-major axis.

  Since the orbit is circular, the eccentricity and longitude of periastron are
  both set to 0.0 in the output.

  Args:
    t0: Time of inferior conjunction.
    period: Orbital period.
    duration: Total transit duration (first to fourth contact), in the same
      units as period.
    planet_radius: Planet radius, in units of stellar radius.
    impact: Impact parameter.
    limb_dark_model: Limb darkening model.
    limb_dark_coeff: List of limb darkening coefficients.

  Returns:
    A batman.TransitParams object.
  """
  # Re-parametrize to the batman parameters.
  semi_major_axis = compute_semi_major_axis(period, duration, planet_radius,
                                            impact)
  inclination = compute_inclination(impact, semi_major_axis)

  params = batman.TransitParams()
  params.t0 = t0
  params.per = period
  params.rp = planet_radius
  params.a = semi_major_axis
  params.inc = inclination
  params.ecc = 0.0  # Assuming circular orbit.
  params.w = 0.0  # Assuming circular orbit.
  params.limb_dark = limb_dark_model
  params.u = limb_dark_coeff if limb_dark_coeff else []

  return params


def transit_model(time,
                  params,
                  supersample_factor=10,
                  exp_time=DEFAULT_EXPOSURE_TIME):
  """Evaluates a transit model at the specified time points."""
  m = batman.TransitModel(
      params, time, supersample_factor=supersample_factor, exp_time=exp_time)
  model_flux = m.light_curve(params)

  return model_flux


# TODO(cshallue): these might be unnecessary; curve_fit might take kwargs that
# are passed along to the function being fit.
def transit_model_uniform(time, t0, period, duration, planet_radius, impact):
  """Evaluates a transit model assuming uniform limb darkening."""
  params = transit_params(t0, period, duration, planet_radius, impact)
  return transit_model(time, params)


def transit_model_quad(time, t0, period, duration, planet_radius, impact, u1,
                       u2):
  """Evaluates a transit model assuming quadratic limb darkening."""
  params = transit_params(t0, period, duration, planet_radius, impact,
                          "quadratic", [u1, u2])
  return transit_model(time, params)


def transit_model_nonlinear(time, t0, period, duration, planet_radius, impact,
                            u1, u2, u3, u4):
  """Evaluates a transit model assuming nonlinear limb darkening."""
  params = transit_params(t0, period, duration, planet_radius, impact,
                          "nonlinear", [u1, u2, u3, u4])
  return transit_model(time, params)


_ParameterSpec = collections.namedtuple("_ParameterSpec",
                                        ["name", "init", "lower", "upper"])


def fit_transit_model(time, flux, t0, period, duration, planet_radius, impact):
  """Fits a transit model with the specified initial parameters."""
  # Set up parameter initial conditions and constraints.
  params = [
      _ParameterSpec("t0", init=t0, lower=0.0, upper=np.inf),
      _ParameterSpec("period", init=period, lower=0.0, upper=np.inf),
      _ParameterSpec("duration", init=duration, lower=0.0, upper=np.inf),
      _ParameterSpec(
          "planet_radius", init=planet_radius, lower=0.0, upper=np.inf),
      _ParameterSpec("impact", init=impact, lower=0.0, upper=1.0),
      _ParameterSpec("u1", init=0.4, lower=0.0, upper=0.5),
      _ParameterSpec("u2", init=0.25, lower=-0.5, upper=0.5),
  ]
  param_names, initial_params, lower_bounds, upper_bounds = zip(*params)

  # Fit transit model.
  # NOTE: Raises RuntimeError if optimal parameters are not found.
  fitted_params = scipy.optimize.curve_fit(
      transit_model_quad,
      time,
      flux,
      p0=initial_params,
      bounds=(lower_bounds, upper_bounds))[0]

  return dict(zip(param_names, fitted_params))


def fit_transit_parameters(time, flux, t0, period, duration, planet_radius):
  """Fits a transit model by sampling initial impact parameters."""
  best_rmse = np.inf
  best_params = None
  for impact in np.linspace(0.01, 0.9, 4):
    fitted_params = fit_transit_model(time, flux, t0, period, duration,
                                      planet_radius, impact)
    fitted_flux = transit_model_quad(time, **fitted_params)
    rmse = np.mean(np.square(flux - fitted_flux))
    if rmse < best_rmse:
      best_rmse = rmse
      best_params = fitted_params

  return best_params
