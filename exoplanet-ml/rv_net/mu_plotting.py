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

def mu_plotting(df, lists, names):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    for i in np.arange(0, len(lists)):
        plt.plot(df["BJD"], df[lists[i]], ".", label=names[i])
        plt.legend()


if __name__ == '__main__':
    mu_plotting()
