import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
from astropy.io.fits import getheader
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import mpyfit
from astropy.io import fits
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import csv
import pickle
import math as m
import pandas as pd
import os


# Create a master shifting function with a lot of flexibility
# Specifically, is able to:
#       - shift to the same reference frame (option, can be disabled)
#       - remove a specific planet signal (option, can be disabled)
#       - inject a specific planet signal (option, can be disabled)
#       - shift to zero or median (both options)

def master_shifting_planet(bjd, ccfBary, rvh,
                           ref_frame_shift,  # "off" or a specific value in km/s
                           removed_planet_rvs,  # array of rv values for planet signal in km/s OR "NULL"
                           injected_planet_params,  # array of amplitude (km/s), 2pi/period, phase
                           zero_or_median):  # "zero" or "median"
    number_of_ccfs = len(ccfBary)

    # HARPS direct data lists
    BJD_list = []
    og_ccf_list = []
    rv_from_HARPS_list = []
    v_rad_raw_list = []

    # injected planet list
    planet_signal_list = []

    # mpyfit lists
    mu_og_list = []
    mu_jup_list = []
    mu_planet_list = []
    mu_zero_list = []
    CCF_normalized_list = []

    # CCF lists
    compiled_ccf_list = []
    jup_shifted_CCF_data_list = []
    planet_shifted_CCF_data_list = []
    shifted_CCF_list = []
    final_ccf_list = []

    def planet_signal(x):
        return injected_planet_params[0] * np.sin(injected_planet_params[1] * x + injected_planet_params[2])

    spline_method = 'quadratic'
    for i in range(0, number_of_ccfs):
        day_of_observation = bjd[i]
        BJD_list.append(day_of_observation)

        # extracts the CCF data and rv from fits
        CCF_data = ccfBary[i]
        og_ccf_list.append(CCF_data)
        rv_from_HARPS = rvh[i]  # - bsrv[i]
        rv_from_HARPS_list.append(rv_from_HARPS)

        # Finds the local minima using a Gaussian fit
        # Define the actual function where     A = p[0], mu = p[1], sigma = p[2], c = p[3]
        def gauss(x, p):
            return -p[0] * np.exp(-(x - p[1]) ** 2 / (2. * p[2] ** 2)) + p[3]

        # A simple minimization function:
        def least(p, args):
            x, y = args
            return gauss(x, p) - y

        parinfo = [{'fixed': False, 'step': 1e-6},
                   {'fixed': False, 'step': 1e-4},
                   {'fixed': False, 'step': 1e-14},
                   {'fixed': False, 'step': 1e-9}]

        # no_shift fit
        rv_data = np.linspace(-20, 20, 161)
        p_no_shifted = [1., 0.1, 1., 0.5]
        pfit_no_shift, results_no_shift = mpyfit.fit(least, p_no_shifted, (rv_data, CCF_data), parinfo)
        mu_og = pfit_no_shift[1]
        mu_og_list.append(mu_og)
        compiled_ccf_list.append(CCF_data)

        # Add in reference frame shift

        if removed_planet_rvs[0] != "NULL":
            jupiter_shift = removed_planet_rvs[i]
            v_rad_raw = rvh[i] + removed_planet_rvs[i]
            v_rad_raw_list.append(v_rad_raw)

            # planet removal shift
            rv_data_jupiter_shift = rv_data + jupiter_shift  # minus sign
            f_jup = interp1d(rv_data_jupiter_shift, CCF_data, kind=spline_method, fill_value='extrapolate')
            jupiter_shifted_CCF_data = f_jup(rv_data)
            jup_shifted_CCF_data_list.append(jupiter_shifted_CCF_data)
            compiled_ccf_list.append(jupiter_shifted_CCF_data)

            # fits the shifted by jupiter data
            p_shifted_jup = [1., 0.1 + jupiter_shift, 1., 0.5]
            pfit_jup, results_jup = mpyfit.fit(least, p_shifted_jup, (rv_data, jupiter_shifted_CCF_data), parinfo)
            m = pfit_jup[1]
            mu_jup_list.append(m)

        if injected_planet_params[0] != "NULL":
            # inject a planet (k=0.3 m/s, p = 365.24d)
            ccf_to_use = compiled_ccf_list[len(compiled_ccf_list) - 1]

            bjd_array = np.asarray(day_of_observation)
            inj_planet_shift = planet_signal(bjd_array)  # km/s
            planet_signal_list.append(inj_planet_shift)
            rv_data_planet_shift = rv_data + inj_planet_shift
            f_planet = interp1d(rv_data_planet_shift, ccf_to_use, kind='cubic', fill_value='extrapolate')
            planet_shifted_CCF_data = f_planet(rv_data)
            planet_shifted_CCF_data_list.append(planet_shifted_CCF_data)
            compiled_ccf_list.append(planet_shifted_CCF_data)

            # fits the shifted by planet data
            p_shifted_planet = [1., 0.1 + inj_planet_shift, 1., 0.5]
            pfit_planet, results_planet = mpyfit.fit(least, p_shifted_planet, (rv_data, planet_shifted_CCF_data),
                                                     parinfo)

            m = pfit_planet[1]
            mu_planet_list.append(m)

            if zero_or_median == "zero":
                # Shift to zero, after planet shift
                ccf_to_use = compiled_ccf_list[len(compiled_ccf_list) - 1]

                shift_to_zero = -(rv_from_HARPS + inj_planet_shift)
                rv_data_shifted = rv_data + shift_to_zero

                f = interp1d(rv_data_shifted, ccf_to_use, kind='cubic', fill_value='extrapolate')
                shifted_CCF_data = f(rv_data)
                shifted_CCF_list.append(shifted_CCF_data)
                compiled_ccf_list.append(shifted_CCF_data)

                # fits the shifted data
                p_shifted = [1., 0.1 - shift_to_zero, 1., 0.5]
                pfit, results = mpyfit.fit(least, p_shifted, (rv_data, shifted_CCF_data), parinfo)
                m_zero = pfit[1]
                mu_zero_list.append(m_zero)  # -0.1)
            else:
                # Shift to median, after planet shift
                ccf_to_use = compiled_ccf_list[len(compiled_ccf_list) - 1]

                shift_to_zero = ((np.mean(rvh) - rv_from_HARPS) - inj_planet_shift)
                rv_data_shifted = rv_data + shift_to_zero

                f = interp1d(rv_data_shifted, ccf_to_use, kind='cubic', fill_value='extrapolate')
                shifted_CCF_data = f(rv_data)
                shifted_CCF_list.append(shifted_CCF_data)
                compiled_ccf_list.append(shifted_CCF_data)

                # fits the shifted data
                p_shifted = [1., 0.1 - shift_to_zero, 1., 0.5]
                pfit, results = mpyfit.fit(least, p_shifted, (rv_data, shifted_CCF_data), parinfo)
                m_zero = pfit[1]
                mu_zero_list.append(m_zero)  # -0.1)
        else:
            if zero_or_median == "zero":
                # Shift to zero
                ccf_to_use = compiled_ccf_list[len(compiled_ccf_list) - 1]

                shift_to_zero = -(rv_from_HARPS)
                rv_data_shifted = rv_data + shift_to_zero

                f = interp1d(rv_data_shifted, ccf_to_use, kind='cubic', fill_value='extrapolate')
                shifted_CCF_data = f(rv_data)
                shifted_CCF_list.append(shifted_CCF_data)
                compiled_ccf_list.append(shifted_CCF_data)

                # fits the shifted data
                p_shifted = [1., 0.1 - shift_to_zero, 1., 0.5]
                pfit, results = mpyfit.fit(least, p_shifted, (rv_data, shifted_CCF_data), parinfo)
                m_zero = pfit[1]
                mu_zero_list.append(m_zero)  # -0.1)
            else:  # shifted to median instead
                ccf_to_use = compiled_ccf_list[len(compiled_ccf_list) - 1]
                shift_to_median = (np.mean(rvh) - rv_from_HARPS)
                rv_data_shifted = rv_data + shift_to_median

                f = interp1d(rv_data_shifted, ccf_to_use, kind='cubic', fill_value='extrapolate')
                shifted_CCF_data = f(rv_data)
                shifted_CCF_list.append(shifted_CCF_data)
                compiled_ccf_list.append(shifted_CCF_data)

                # fits the shifted data
                p_shifted = [1., 0.1 - shift_to_median, 1., 0.5]
                pfit, results = mpyfit.fit(least, p_shifted, (rv_data, shifted_CCF_data), parinfo)
                m_zero = pfit[1]
                mu_zero_list.append(m_zero)  # -0.1)
        ccf_to_use = compiled_ccf_list[len(compiled_ccf_list) - 1]
        final_ccf_list.append(ccf_to_use)

        # normalize the CCFs
        x_left = ccf_to_use[0:40]
        x_right = ccf_to_use[121:161]
        x_norm_range = list(x_left) + list(x_right)
        CCF_normalized = ccf_to_use * (1 / np.mean(x_norm_range))
        CCF_normalized_list.append(CCF_normalized)

    # Create a dataframe
    d = {'BJD': BJD_list,
         'vrad_star': rvh,
         'vrad_plan_star': rvh + planet_signal_list,
         'og_ccf_list': og_ccf_list,
         'jup_shifted_CCF_data_list': jup_shifted_CCF_data_list,
         'planet_shifted_CCF_data_list': planet_shifted_CCF_data_list,
         'zero_shifted_CCF_list': shifted_CCF_list,
         'CCF_normalized_list': CCF_normalized_list,
         'mu_og_list': mu_og_list,
         'mu_jup_list': mu_jup_list,
         'mu_planet_list': mu_planet_list,
         'mu_zero_list': mu_zero_list
         }
    df = pd.DataFrame(data=d)

    return df


if __name__ == '__main__':
    master_shifting_planet()
