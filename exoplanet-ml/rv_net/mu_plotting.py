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

def mu_plotting (df, names):
    fig, ax = plt.subplots(1,1, figsize=(8, 4))
    plt.plot(df["BJD"], df["mu_og_list"],".", label = names[0])
    plt.plot(df["BJD"], df["mu_jup_list"], ".",label = names[1])
    plt.plot(df["BJD"], df["mu_planet_list"], ".",label = names[2])
    plt.plot(df["BJD"], df["mu_zero_list"], ".", color="k",label = names[3])
    plt.legend()
    
    
if __name__ == '__main__':
    mu_plotting()
