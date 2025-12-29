# crossCalibrateSXR.py
"""
This program was written to find a cross-calibration correction factor between
the four SXR arrys on DIII-D.

This program assumes that the radiation is axisymmetric during the current quench,
therefore it will fit a radDist to the SXR data, assuming axisymmetry,
and it will find the cross-calibration factor that minimizes the difference
between the data and the radDist.

The radDists are assumed to be in the inputs/{tokamakName}/radDists/ directory.

This program will fit all of the radDists using a variable to solve for each array,
it will then adjust the scaling factor based on the best fit variables for each array,
then repeat again until the scaling factors don't vary that much between runs.

The goal is to have all of the multiplcation factors from the fitting the same for each array.

Order of operations:
1) useGrabSXRData.py to grab and save the SXR data
2) use this program to find the cross-calibration factors
3) put this factor in your specific run setting (e.g. 184407_runConfig.yaml)

NOTE: You will need pre-generated elongated ring radDists for this to work.

# Written by Jeffrey Herfindal, Aug. 26, 2025
"""


import os
import sys
import numpy as np

# --- Path to where the Globals.py file is!
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import main.Emis3D as Emis3D
from main.Globals import *
from lmfit import report_fit

# --- Values to change ---
shotNumber = 184407
path_to_config = os.path.join(os.path.dirname(__file__), "SXR_crossCalib_Config.yaml")
time_start = 2124.0
dt = 0.3


# -------------------------
# --- Start of program ---
# -------------------------
scale = None
scaleArray = None
emis = None
for ii in range(10):
    emis = Emis3D.Emis3D(
        tokamakName="DIII-D",
        runConfigName=None,
        initialize=False,
        verbose=False,
    )
    emis._load_config_file(
        "DIII-D", "SXR_crossCalib_Config.yaml", pathFileName=path_to_config
    )

    # --- Initilize cross-calibration scaling dictionary
    if scale is None and emis.info is not None:
        scale = {}
        for bolo_ in list(emis.info["BOLOMETERS"].keys()):
            scale[bolo_] = 1

    #  --- Apply new scaling to the data
    if emis.info is not None and scale is not None:
        for bolo_ in emis.info["BOLOMETERS"]:
            emis.info["BOLOMETERS"][bolo_]["scalingFactor"] = scale[bolo_]

    # --- Preform the fits
    emis._load_bolometer_data()
    emis._create_master_channel_order()
    emis._load_radDists()
    emis._prepare_fits(evalTime=float(time_start), crossCalib=True)
    emis._minimize_radDists(evalTime=float(time_start), crossCalib=True)
    emis._post_process_fit_arrangement(evalTime=float(time_start), crossCalib=True)

    # --- Find the new multiplication factors in the best fit
    fit_ = emis.bestFits[float(time_start)]["fit"].params.valuesdict()

    # --- Find the scale factor relative to the highest value
    if scaleArray is None and emis.info is not None:
        max_ = 0
        for val in fit_:
            if val in list(emis.info["BOLOMETERS"].keys()) and fit_[val] > max_:
                scaleArray = val

    # --- Update the scale factors
    diff_ = 0
    if scale is not None:
        for bolo_ in scale:
            scale[bolo_] *= fit_[scaleArray] / fit_[bolo_]
            diff_ += np.abs(fit_[scaleArray] - fit_[bolo_])

    # --- Print the current scale factors
    print("-" * 50)
    print(f"After run {ii + 1}, the scale factors are:\n")
    if scale is not None:
        for bolo_ in scale:
            print(f"{bolo_}: {scale[bolo_]:.2f}")
    aic_ = emis.bestFits[float(time_start)]["fit"].aic.item()
    bic_ = emis.bestFits[float(time_start)]["fit"].bic.item()
    chisq = emis.bestFits[float(time_start)]["fit"].chisqr.item()
    print("\n")
    print(report_fit(emis.bestFits[float(time_start)]["fit"]))
    print("-" * 50)

    # --- Determine if the new scale factors are within limits
    if diff_ < 0.2:
        print("Cross-calibration done!")
        break

if emis is not None:
    # --- Print the final scale factors
    print("*" * 50)
    print("*" * 50)
    print("The final scale factors are:\n")
    if scale is not None:
        for bolo_ in scale:
            print(f"{bolo_}: {scale[bolo_]:.2f}")
    aic_ = emis.bestFits[float(time_start)]["fit"].aic.item()
    bic_ = emis.bestFits[float(time_start)]["fit"].bic.item()
    chisq = emis.bestFits[float(time_start)]["fit"].chisqr.item()
    print("\n")
    print(report_fit(emis.bestFits[float(time_start)]["fit"]))
    print("*" * 50)
    print("*" * 50)
    emis._plot_bestFit(evalTime=float(time_start))
