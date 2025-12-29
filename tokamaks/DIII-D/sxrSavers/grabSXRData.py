# grab_sxr_store_data.py
#
# This program will grab the SXR data, correct for offset, and filter the data.
# It ment to be run several times to zero out and correct bad channels prior to
# saving the data in inputs/DIII-D/sxrData/
#
# Written by Jeffrey Herfindal, Feb. 26, 2024
#
# TODO: Add ability to increase half of the array since each bolometer is two arrays
# Check to see why the badChannels thing is not working

import os
import sys

# --- Path to where the Globals.py file is!
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
import matplotlib.pyplot as plt
import numpy as np
import radiation
from uncertainties import unumpy as unp

# --- Values to change ---
shotNumber = 184407
loadData = False
arrayName = "SX90MF"  # Supported arrays: SX45F, SX90PF, SX90MF, DISRADU
badChannels = []  # Zero out these channels before saving
invertedChannels = []  # Channels to be inverted
offsetWindow = [4000, 6000]  # Enter an offset window, optional
filterWindowLength = 41
filterPolyOrder = 3
save_data = False
truncateTimeRange = [2115, 2134]

# --- Multiplcation factors for the sub arrays
"""
Each of the bolometers within DIII-D consist of two diode arrays, this
multiplaction factor is in place to account for inconsistencies between
the arrays. The genral procedure is to look at a time segment that has signal
between the middle channels and see if the signals look resonable. Change the 
average_time to and look at the signal vs channel plot

First = The first half of the channels (e.g. 1 through 15)
Second = the second half of the channels (e.g. 16 through 32)
"""
mult_factor_first = 1.0
mult_factor_second = 1.0


# --- Plotting commands
plot_overview = True  # Plot raw and processed data
plot_zoom = False  # Zoom plot overview over the timeRange window


# --- If you want to see what values the program will use:
plot_average_window = True
average_times = np.arange(2121, 2125, 0.5)
average_dt = 0.1

# -------------------------
# --- Start of program ---
# -------------------------

# --- Grab the data
runner = False
if not loadData:
    runner = True
data = radiation.SXRSignal(
    shotNumber=shotNumber,
    arrayName=arrayName,
    offsetWindow=offsetWindow,
    filterWindowLength=filterWindowLength,
    filterPolyOrder=filterPolyOrder,
    badChannels=badChannels,
    invertedChannels=invertedChannels,
    truncateTimeRange=truncateTimeRange,
    runner=runner,
)

if loadData:
    data.load_data()
    data._num_channels()

# --- Apply the calibration factor
num_channels = data.info["numChannels"]
if "PROCESSED_SIGNAL" in data.data:
    for ii in range(int(num_channels / 2.0)):
        data.data["PROCESSED_SIGNAL"][ii, :] *= mult_factor_first
    for ii in range(int(num_channels / 2.0), num_channels):
        data.data["PROCESSED_SIGNAL"][ii, :] *= mult_factor_second

# --- Calibrate the data
data.calibrate_data()

if save_data:
    data.save_data()


# --- Plot the data overview
if plot_overview:
    f = plt.figure(figsize=(16, 10))

    for ii in range(len(data.data["channelOrder"])):
        ax = f.add_subplot(5, 7, ii + 1)

        ax.plot(
            data.get_time(raw=True),
            data.get_raw_data()[ii, :],
            color="grey",
            label="raw",
        )
        ax.plot(
            data.get_time(), data.get_observed()[ii, :], color="teal", label="filtered"
        )
        ax.text(
            0.01,
            0.98,
            f"{data.data['channelOrder'][ii]}",
            va="top",
            ha="left",
            transform=ax.transAxes,
        )

        ax.hlines(0, 4000, 0, color="red", linewidth=2.0)
        ax.legend(loc="upper right")
        if plot_zoom:
            ax.set_xlim(truncateTimeRange[0], truncateTimeRange[1])
    plt.tight_layout()


if plot_average_window:
    f = plt.figure()
    ax = f.add_subplot(111)
    for average_time in average_times:
        d_ = [0] * data.get_calibrated().shape[0]
        d_e = [0] * data.get_calibrated().shape[0]
        loc_s = np.abs(data.get_time() - (average_time - average_dt)).argmin()
        loc_e = np.abs(data.get_time() - (average_time + average_dt)).argmin()

        for ii, da in enumerate(range(data.info["numChannels"])):
            if np.mean(data.get_calibrated()[ii, loc_s:loc_e]) == 0:
                data.get_calibrated()[ii, loc_s:loc_e] = np.nan

            temp_ = unp.uarray(
                data.get_calibrated()[ii, loc_s:loc_e],
                data.data["DATA_CALIBRATED_ERROR"][ii, loc_s:loc_e],
            )

            d_[ii] = unp.nominal_values(np.mean(temp_))
            d_e[ii] = unp.std_devs(np.mean(temp_))

        ax.errorbar(
            np.arange(1, len(d_) + 1),
            d_,
            yerr=d_e,
            marker="s",
            label=f"{average_time:.2f} ms",
        )
    ax.set_xlabel("Channel")
    ax.set_ylabel("Calibrated Signal (W/(m2 sr)")
    ax.set_title(f"{arrayName} {shotNumber}")
    ax.legend()

if plot_overview or plot_average_window:
    plt.show()
