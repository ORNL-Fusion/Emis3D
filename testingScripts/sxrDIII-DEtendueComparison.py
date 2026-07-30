# sxrDIII-DEtendueComparison.py
"""
Calculates the etendue with Emis3D and compares it to the measured values

NOTE: If you are getting this error:
'Bolometer' object has no attribute 'etendues_analytic'

It is because you have the ETENDUE field in the bolometer configuration files,
so the program skips ray-tracing the etendue's.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np
from main.Tokamak import Tokamak

# import tokamaks.DIII_D.sxrSavers.Util_SXR as Util_SXR
from pathlib import Path

module_path = str(Path("../tokamaks/DIII-D/sxrSavers/").resolve())
sys.path.insert(0, module_path)
import Util_SXR  # type: ignore
from raysect.optical import AbsorbingSurface, NullMaterial  # type: ignore

tok = Tokamak(
    tokamakName="DIII-D",
    mode="Build",
    reflections=False,
    loadBolometers=True,
)


tok.calc_etendues()


info = Util_SXR._get_calib_info(184407, ArrayName="SXR45")
etendue_ = {}
if tok.info is not None:
    for b_ in tok.info["Bolometer Groups"]:
        if b_ == "SX90PF":
            n_ = "SX90RP1F"
        elif b_ == "SX90MF":
            n_ = "SX90RM1F"
        elif b_ == "SX45F":
            n_ = "SXR45"
        else:
            n_ = b_
        etendue_[b_] = Util_SXR.get_calibration(184407, ArrayName=n_)

# --- Combine the calculated etendues
e_ = {}
for b_ in tok.bolometers:
    chan = None
    if b_.info["GROUP_NAME"] not in e_:
        e_[b_.info["GROUP_NAME"]] = {}
        for item in [
            "chan",
            "etendue",
            "etendue_error",
            "etendue_analytic",
            "etendue_analytic_error",
        ]:
            e_[b_.info["GROUP_NAME"]][item] = []

    if b_.name == "SX90PF_UP":
        chan = np.arange(17, 33)
    elif b_.name == "SX90PF_DOWN":
        chan = np.arange(1, 17)
    elif b_.name == "SX90MF_UP":
        chan = np.arange(17, 33)
    elif b_.name == "SX90MF_DOWN":
        chan = np.arange(1, 17)
    elif b_.name == "DISRADU_UP":
        chan = np.arange(16, 31)
    elif b_.name == "DISRADU_DOWN":
        chan = np.arange(1, 16)
    elif b_.name == "SX45F_UP":
        chan = np.arange(8, 13)
    elif b_.name == "SX45F_DOWN":
        chan = np.arange(1, 8)

    if chan is not None:
        e_[b_.info["GROUP_NAME"]]["chan"] += chan.tolist()
        e_[b_.info["GROUP_NAME"]]["etendue"] += b_.etendues
        e_[b_.info["GROUP_NAME"]]["etendue_error"] += b_.etendues_error
        e_[b_.info["GROUP_NAME"]]["etendue_analytic"] += b_.etendues_analytic
        e_[b_.info["GROUP_NAME"]][
            "etendue_analytic_error"
        ] += b_.etendues_analytic_error


plt.ion()


# --- Plot each individual bolometer
if tok.info is not None:
    boloGroups = tok.info["Bolometer Groups"]
    bolometers = tok.bolometers

    num_columns = len(boloGroups)
    num_rows = 2
    f = plt.figure(figsize=(15, 8))

    # --- Loop over each bolometer group
    plot_count = 0
    for ii, boloGroup in enumerate(boloGroups):
        plot_count += 1
        f_ = f.add_subplot(num_rows, num_columns, plot_count)
        tok._plot_first_wall(f_)
        tok._plot_bolometers(
            f_,
            boloGroupName=boloGroup,
            plot_chord_info=True,
            plot_etendue=["SX90PF07", "SX90MF07", "DISRADU07", "SX45F07"],
            legend=True,
        )

    # --- Plot the etendues
    for ii, boloGroup in enumerate(boloGroups):
        plot_count += 1
        f_ = f.add_subplot(num_rows, num_columns, plot_count)
        chan = np.arange(1, etendue_[boloGroup]["ETENDUE"].shape[0] + 1)
        eten_ = etendue_[boloGroup]["ETENDUE"]
        f_.scatter(chan, eten_, marker="s", color="black", label="Measured")
        norm = np.nanmax(eten_) / np.nanmax(np.array(e_[boloGroup]["etendue"]))
        print(f"{boloGroup} Normalization factor: {norm:.2e}, calculated / measured")
        f_.errorbar(
            e_[boloGroup]["chan"],
            np.array(e_[boloGroup]["etendue"]),
            yerr=np.array(np.array(e_[boloGroup]["etendue_error"])),
            fmt="-o",
            color="steelblue",  # Color of the data line/markers
            ecolor="gray",  # Color of the error bars
            elinewidth=2,  # Line width of the error bars
            capsize=5,  # Cap size
            capthick=2,  # Cap thickness
            ms=7,
            label="Raysect ray-traced",
            linestyle="none",
        )
        f_.scatter(
            e_[boloGroup]["chan"],
            np.array(e_[boloGroup]["etendue_analytic"]),
            color="tab:red",
            label="Raysect Analytic",
            marker="^",
        )
        f_.set_title(boloGroup)
        f_.legend()

    plt.tight_layout()
    plt.show()
