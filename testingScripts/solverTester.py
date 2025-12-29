#solverTester.py
"""
This program was written to test out the emis3D solver routine. The main point
is to load a known radDist, determine the correct scaling coefficients, use the 
Util.emis3d.synthetic_after_fit function to scale the synthetic data, and then compare
the scaled synthetic data to the original data.

The program will then also step through the emis3D solver, using the residual function to
see how close the fits are. 

Written by Jeffrey Herfindal, August 22, 2025.
"""


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np
import main.Util_radDist as Util_radDist
from main.Globals import *
from main.Tokamak import Tokamak
from main.Util import config_loader
from raysect.core import Point2D
import main.Emis3D as Emis3D
from main.Util_emis3D import synthetic_after_fit, residual, scale_exp

evalTime = 2123
params_dict = {
"a_injection_location_0_clockwise":1.0,
"a_injection_location_0_counterClock":2.0,
"b_injection_location_0_clockwise":1.0,
"b_injection_location_0_counterClock": 0.0,
}

def point3d_to_rz(point):
    return Point2D(np.hypot(point.x, point.y), point.z)

def pellet_initial_parameters(csp=False):
    """
    Gives the basic SPI trajector parameters
    """
    data = {}
    if csp:
        # --- CSP trajectory parameters
        data["R_OUT"] = 2.35  # 2.249
        data["Z_OUT"] = 0.0

        # SPI breaker tube angle down (degrees)
        data["THETA"] = 0
        data["DISP"] = 2

        # SPI length (arbitrary, can increase or decrease)
        data["LENGTH"] = 0.8

    else:
        # --- SPI trajectory parameters
        data["R_OUT"] = 2.284
        # SPI breaker tube tip Z (outer wall)
        data["Z_OUT"] = 0.6845

        # SPI breaker tube angle down (degrees)
        data["THETA"] = 47.3
        data["DISP"] = 15.0

        # SPI length (arbitrary, can increase or decrease)
        data["LENGTH"] = 0.8

    data["Z_IN"] = data["Z_OUT"] - data["LENGTH"] * np.sin(np.deg2rad(data["THETA"]))
    data["R_IN"] = data["R_OUT"] - data["LENGTH"] * np.cos(np.deg2rad(data["THETA"]))

    # --- Find the upper and lower scatter points
    dz = data["LENGTH"] * np.sin((np.deg2rad(data["THETA"] - data["DISP"])))
    dr = data["LENGTH"] * np.cos((np.deg2rad(data["THETA"] - data["DISP"])))
    data["Z_IN_UPPER"] = data["Z_OUT"] - dz
    data["R_IN_UPPER"] = data["R_OUT"] - dr

    dz = data["LENGTH"] * np.sin((np.deg2rad(data["THETA"] + data["DISP"])))
    dr = data["LENGTH"] * np.cos((np.deg2rad(data["THETA"] + data["DISP"])))
    data["Z_IN_LOWER"] = data["Z_OUT"] - dz
    data["R_IN_LOWER"] = data["R_OUT"] - dr

    return data

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Step 1: Create a radDist
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
tokamakName = "DIII-D"
configFileName = "184407_injectionLocation_225/helical_config.yaml"
elongation = 1.0
polSigma = 0.1
rzvalues = [2.113, 0.5]  # [R,Z] values for the center of the radDist
rzvalues = [2.107, 0.496]  # [R,Z] values for the center of the radDist
# --- Group the bolometers
bolometerNames = [
    ["SX90PF_UP", "SX90PF_DOWN"],
    ["SX90MF_UP", "SX90MF_DOWN"],
    ["DISRADU_UP", "DISRADU_DOWN"],
    # ["SX45F_UP", "SX45F_DOWN"],
]
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
)

rzArray = Util_radDist.callRZGridTokamak(
    tok,
    numRgrid=config["GRID"]["NumRStartGrid"],
    numZgrid=config["GRID"]["NumZStartGrid"],
)
rzArray[0] = [rzvalues[0], rzvalues[1]]

config["polSigma"] = polSigma
config["elongation"] = elongation
config["saveRunsDirectoryName"] = "solverTesting"
arg_list = [(val, config) for val in rzArray]

arg_list = arg_list[0]


# --- Delete old radDist directory if it exists
radDist_dir = os.path.join(EMIS3D_INPUTS_DIRECTORY, tokamakName
                          , "radDists", "solverTesting")
if os.path.exists(radDist_dir):
    import shutil
    shutil.rmtree(radDist_dir)
# --- This will not work after I am done testing since the definition will not return the radDist
hel = Util_radDist.radDist_Helical_parallel(arg_list)



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Step 2: Load the data in emis3D, create synthetic signal based off known parameters
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

t = Emis3D.Emis3D(tokamakName="DIII-D", runConfigName="184407/184407_runConfig.yaml")
if t.info is not None:
    t.info['radDistDirectories'] = ['solverTesting']
t._load_radDists()
t._load_bolometer_data()

t._prepare_data_for_fit(evalTime=float(evalTime))
t._prepare_synthetic_for_fits(evalTime=float(evalTime))
ii = 0

data_dict = t.fitData[evalTime]
synthetic_dict = t.fitSynthetic["injection_location_0"][ii]["data"].copy()
par = t.fitSynthetic["injection_location_0"][ii]["params"]

# --- Update with known parameters
for key in params_dict:
    par[key].set(value=params_dict[key])

if t.info is not None:
    scale_def = t.info["scale_def"]
else:
    scale_def = "linear"

data_manual = {}
data_manual = synthetic_after_fit(par, synthetic_dict, scale_def=scale_def)

# --- Arrange each set in a dictionary for easy plotting
data_manual_dict = {}
synthetic_manual_dict = {}
for emissionName in data_manual:
    data_manual_dict[emissionName] = {}
    synthetic_manual_dict[emissionName] = {}
    for ii in range(len(t.channel_order["channel_list"])):
        map_ = dict(
            zip(t.channel_order["channel_list"][ii], data_manual[emissionName][ii].copy())
        )
        data_manual_dict[emissionName].update(map_)

        map_2 = dict(
            zip(t.channel_order["channel_list"][ii], synthetic_dict[emissionName]['data'][ii].copy())
        )
        synthetic_manual_dict[emissionName].update(map_2)

# --- Delete the initial emis3D object since it is no longer needed
del t

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Step 3: Check the residual function with the current fit parameters
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
res_manual = residual(par, data_dict, synthetic_dict, scale_def=scale_def)




# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Step 4: Perform a minimization using the emis3D solver
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
t2 = Emis3D.Emis3D(tokamakName="DIII-D", runConfigName="184407/184407_runConfig.yaml")
t2._load_radDists()
t2._load_bolometer_data()
t2._prepare_fits(evalTime=float(evalTime))




# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Step 5: Plot the results
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# --- Find the location of each bolometer
bolo_tokamak = []
if hel is not None and hasattr(hel, "tokamak") and hasattr(hel.tokamak, "bolometers"):
    for bolo in hel.tokamak.bolometers: # type: ignore
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
            for foil in hel.tokamak.bolometers[indx_].bolometer_camera: # type: ignore
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
                int(hel.tokamak.bolometers[indx_plot].info["CAMERA_POSITION_R_Z_PHI"][2])
            ),
            ax=f_top,
        )


    # --- Plot SPI injection location
    f_top = f.add_subplot(2, num_rows, len(bolometerNames) + 1)
    tok._plot_first_wall(f_top)
    hel.plotCrossSection(phi=np.deg2rad(config["injectionLocation"]), ax=f_top)
    spi_path = pellet_initial_parameters(csp=False)
    f_top.plot(
        [spi_path["R_IN"], spi_path["R_OUT"]],
        [spi_path["Z_IN"], spi_path["Z_OUT"]],
        "-r",
    )
    f_top.plot(
        [spi_path["R_IN_UPPER"], spi_path["R_OUT"]],
        [spi_path["Z_IN_UPPER"], spi_path["Z_OUT"]],
        "-r",
    )
    f_top.plot(
        [spi_path["R_IN_LOWER"], spi_path["R_OUT"]],
        [spi_path["Z_IN_LOWER"], spi_path["Z_OUT"]],
        "-r",
    )
    f_top.set_title("Injection Location")

    # --- Plot the LCFS
    f_top.plot(tok.gfile.rbbbs, tok.gfile.zbbbs, "-k", linewidth=1)

    # --- Plot the observed emissivities

    colors = ["black", "purple"]

    for ii, boloGroup in enumerate(bolometerNames):
        plot_observed = True
        f_top = f.add_subplot(2, num_rows, ii + 6)
        for qq, emissionName in enumerate(hel.info["emissionNames"]):

            # --- Manual synthetic data
            data_observed = []
            data_observed_err = []
            data_synthetic = []
            data_synthetic_manual = []
            # --- Group the data
            data_ = []
            chan_ = []
            for jj, bolo in enumerate(boloGroup):
                indx_ = bolo_tokamak.index(bolo)
                ch_tags = hel.tokamak.bolometers[indx_].info["CHANNEL_TAGS"]
                c_ = []
                for ch in ch_tags: # type: ignore
                    c_.append(int(ch[-2:]))
                    data_synthetic.append(synthetic_manual_dict[emissionName][ch])
                    data_synthetic_manual.append(data_manual_dict[emissionName][ch])
                    data_observed.append(data_dict['dataMap'][ch])
                    data_observed_err.append(data_dict['dataMap'][ch] * 0.3)

                data_ += hel.data[hel.info["units"]][emissionName][bolo]
                chan_ += c_

            # --- Sort the channel list in ascending order
            inds = np.array(chan_).argsort()
            f_top.plot(
                np.array(chan_)[inds],
                np.array(data_synthetic)[inds],
                color=colors[qq],
                label=f"{emissionName} synthetic data",
            )
            f_top.plot(
                np.array(chan_)[inds],
                np.array(data_synthetic_manual)[inds],
                color=colors[qq],
                linestyle="dashed",
                label=f"{emissionName} (manual synthetic)",
            )

            if plot_observed:
                f_top.errorbar(
                    np.array(chan_)[inds],
                    np.array(data_observed)[inds],
                    yerr=np.abs(np.array(data_observed_err))[inds],
                    fmt="o",
                    color='black',
                    label=f"{emissionName} observed data",
                )
        plot_observed = False

        f_top.legend(fontsize=8)
        f_top.set_ylabel(f"{hel.data['units']}")
        f_top.set_xlabel("channel")

    plt.tight_layout()
    plt.show()
