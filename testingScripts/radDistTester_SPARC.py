# radDistTester_JET.py
"""
This program will group similar SXR arrays, then plot out
the chords, radDist contour plot, and the observed radiation
below.

It is currently specific to SPARC
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_REPO_ROOT))

import numpy as np
import main.Util_radDist as Util_radDist
from main.Globals import EMIS3D_INPUTS_DIRECTORY
from main.Util import config_loader
import matplotlib.pyplot as plt
from os.path import join

tokamakName = "SPARC_FDR"
configFileName = "NIMROD_config.yaml"


# --- Create the radDist using only one point, we don't need to loop over everything
pathFileName = EMIS3D_INPUTS_DIRECTORY / tokamakName / "radDists" / configFileName
config = config_loader(pathFileName)
print(pathFileName)
if config is None:
    raise FileNotFoundError(f"Could not load config file: {pathFileName}")

# for time in np.arange(60, 2880, 60):

if True:
    time = 60
    timestep = f"{time:05d}"  # f"{60:05d}"
    nimrodFile = join(
        EMIS3D_INPUTS_DIRECTORY,
        tokamakName,
        "radDists",
        "NIMROD_data",
        "Nonresonant",
        f"nimrod_Nonresonant_{timestep}.txt",
    )

    # --- Update the configuration file
    # rzArray = np.array([rzvalues[0], rzvalues[1]])

    if "sigma_R_vals" in config:
        config["sigma_R"] = 0.1  # config["sigma_R_vals"][0]
    if "sigma_z_vals" in config:
        config["sigma_z"] = 0.001  # config["sigma_z_vals"][0]
    if "rotationAngles" in config:
        config["rotationAngle"] = 0  # config["rotationAngles"][0]
    # arg_list = (rzArray, config)
    arg_list = (nimrodFile, timestep, config)

    print("Setup complete")
    # --- Create the radDist
    if config["distType"] == "Helical":
        rD = Util_radDist.radDist_Helical_parallel(arg_list, return_result=True)
    elif config["distType"] == "HelicalRing":
        rD = Util_radDist.radDist_HelicalRing_parallel(arg_list, return_result=True)
    elif config["distType"] == "ElongatedRing":
        rD = Util_radDist.radDist_ElongatedRing_parallel(arg_list, return_result=True)
    elif config["distType"] == "SquareTube":
        rD = Util_radDist.radDist_SquareTube_parallel(arg_list, return_result=True)
    elif config["distType"] == "NIMROD":
        rD = Util_radDist.radDist_NIMROD_parallel(arg_list, return_result=True)
    else:
        raise RuntimeError(
            "Please have 'elongatedRing', 'helical', 'HelicalRing', 'NIMROD', or 'SquareTube' in the configFileName"
        )
    print(f"RadDist {timestep} built")


# # --- Plot everything ---
# if rD is not None:
#     fig = rD.plotOverview(plot_etendue=[""], return_figure=True)
#     # Add the pellet scatter
#     if fig is not None:
#         #pellet_scatter(fig.axes[2])
#         plt.show()
