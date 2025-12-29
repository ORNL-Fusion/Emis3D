# radDistTester_DIII-D.py
"""
This program will group similar SXR arrays, then plot out
the chords, radDist contour plot, and the observed radiation
below.

It is currently specific to DIII-D
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np
from raysect.core import Point2D

import main.Util_radDist as Util_radDist
from main.Globals import *
from main.Tokamak import Tokamak
from main.Util import config_loader

tokamakName = "DIII-D"
configFileName = "184407_injectionLocation_225/helical_config.yaml"
elongation = 0.3
polSigma = 0.25
rotationAngle = 30
rzvalues = [2.05135, 0.3508]

# --- Group the bolometers
bolometerNames = [
    ["SX90PF_UP", "SX90PF_DOWN"],
    ["SX90MF_UP", "SX90MF_DOWN"],
    ["DISRADU_UP", "DISRADU_DOWN"],
    # ["SX45F_UP", "SX45F_DOWN"],
]


def point3d_to_rz(point):
    return Point2D(np.hypot(point.x, point.y), point.z)


# --- Start of program!!
pathFileName = os.path.join(
    EMIS3D_INPUTS_DIRECTORY, tokamakName, "radDists", configFileName
)
config = config_loader(pathFileName)
if config is None:
    raise FileNotFoundError(f"Could not load config file: {pathFileName}")

tok = Tokamak(
    tokamakName=tokamakName,
    mode="Build",
    reflections=False,
    eqFileName="g184407.02100",
    loadBolometers=False,
    verbose=True,
)

rzArray = Util_radDist.callRZGridTokamak(
    tok,
    numRgrid=config["GRID"]["NumRStartGrid"],
    numZgrid=config["GRID"]["NumZStartGrid"],
)
rzArray[0] = [rzvalues[0], rzvalues[1]]

config["polSigma"] = polSigma
config["elongation"] = elongation
config["rotationAngle"] = rotationAngle
arg_list = [(val, config) for val in rzArray]

arg_list = arg_list[0]

# --- This will not work after I am done testing since the definition will not return the radDist
hel = Util_radDist.radDist_Helical_parallel_return_radDist(arg_list)
# hel = Util_radDist.radDist_ElongatedRing_parallel(arg_list)


# --- Find the location of each bolometer
bolo_tokamak = []
if hel is not None and hasattr(hel, "tokamak") and hasattr(hel.tokamak, "bolometers"):
    for bolo in hel.tokamak.bolometers:  # type: ignore
        bolo_tokamak.append(bolo.name)
else:
    raise AttributeError("hel or hel.tokamak.bolometers is not properly initialized.")


# --- Plot each individual bolometer
if True:
    num_rows = len(bolometerNames) + 1
    f = plt.figure(figsize=(15, 8))

    for ii, boloGroup in enumerate(bolometerNames):
        f_top = f.add_subplot(2, num_rows, ii + 1)
        tok._plot_first_wall(f_top)
        for bolo in boloGroup:
            indx_ = bolo_tokamak.index(bolo)
            for foil in hel.tokamak.bolometers[indx_].bolometer_camera:  # type: ignore
                slit_centre = foil.slit.centre_point
                slit_centre_rz = point3d_to_rz(slit_centre)
                f_top.plot(slit_centre_rz[0], slit_centre_rz[1], "ko")
                origin, hit, _ = foil.trace_sightline()
                centre_rz = point3d_to_rz(foil.centre_point)
                f_top.plot(centre_rz[0], centre_rz[1], "kx")
                origin_rz = point3d_to_rz(origin)
                hit_rz = point3d_to_rz(hit)
                f_top.plot([origin_rz[0], hit_rz[0]], [origin_rz[1], hit_rz[1]], "k")
                f_top.text(
                    hit_rz[0],
                    hit_rz[1],
                    str(int(foil.name[-2:])),
                    fontsize="10",
                    ha="center",
                    va="center",
                    weight="bold",
                )
        f_top.set_title(boloGroup[0].split("_")[0])
        # --- Plot the radDist
        # Use the first bolometer in the group to get indx_
        indx_plot = bolo_tokamak.index(boloGroup[0])
        hel.plotCrossSection(
            phi=np.deg2rad(
                int(
                    hel.tokamak.bolometers[indx_plot].info["CAMERA_POSITION_R_Z_PHI"][2]
                )
            ),
            ax=f_top,
        )

    # --- Plot SPI injection location
    f_top = f.add_subplot(2, num_rows, len(bolometerNames) + 1)
    tok._plot_first_wall(f_top)
    hel.plotCrossSection(phi=np.deg2rad(config["injectionLocation"]), ax=f_top)
    f_top.set_title("Injection Location")

    # --- Plot the observed emissivities

    colors = ["black", "purple", "blue", "green", "orange", "red"]

    for ii, boloGroup in enumerate(bolometerNames):
        f_top = f.add_subplot(2, num_rows, ii + 6)
        for qq, emissionName in enumerate(hel.info["emissionNames"]):
            # --- Group the data
            data_ = []
            chan_ = []
            for jj, bolo in enumerate(boloGroup):
                indx_ = bolo_tokamak.index(bolo)
                ch_tags = hel.tokamak.bolometers[indx_].info["CHANNEL_TAGS"]
                c_ = []
                for ch in ch_tags:  # type: ignore
                    c_.append(int(ch[-2:]))

                data_ += hel.data[hel.info["units"]][emissionName][bolo]
                chan_ += c_

            # --- Sort the channel list in ascending order
            inds = np.array(chan_).argsort()
            f_top.plot(
                np.array(chan_)[inds],
                np.array(data_)[inds],
                color=colors[qq],
                label=emissionName,
            )

        f_top.legend()
        f_top.set_ylabel(f"{hel.data['units']}")
        f_top.set_xlabel("channel")

    plt.tight_layout()
    plt.show()
