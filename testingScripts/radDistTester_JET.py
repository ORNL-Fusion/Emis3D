# radDistTester_JET.py
"""
This program will group similar SXR arrays, then plot out
the chords, radDist contour plot, and the observed radiation
below.

It is currently specific to JET
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_REPO_ROOT))

import numpy as np
import emis3d.Util_radDist as Util_radDist
from emis3d.workspace import get_workspace
from emis3d.Util import config_loader
import matplotlib.pyplot as plt

def pellet_scatter(ax):
    """Gives SPI trajectory range, for plotting"""

    y_in = [0.75, 1.88]
    x_in = [2.23, 3.17]

    y_out = [0.406, 1.88]
    x_out = [2.91, 3.17]

    ax.plot(x_in, y_in, linestyle = 'dashed', color = 'red', linewidth = 2.0)
    ax.plot(x_out, y_out, linestyle = 'dashed', color = 'red', linewidth = 2.0)


tokamakName = "JET"
configFileName = (
    "helical_config.yaml"  # helical_config.yaml, or elongatedRing_config.yaml
)
rzvalues = [3.019, 1.55]
rzvalues = [3.2, 1.3]


# --- Create the radDist using only one point, we don't need to loop over everything
pathFileName = get_workspace().path("raddists", tokamak=tokamakName) / configFileName
config = config_loader(pathFileName)
print(pathFileName)
if config is None:
    raise FileNotFoundError(f"Could not load config file: {pathFileName}")


# --- Update the configuration file
rzArray = np.array([rzvalues[0], rzvalues[1]])

if "sigma_R_vals" in config:
    config["sigma_R"] = 0.1 #config["sigma_R_vals"][0]
if "sigma_z_vals" in config:
    config["sigma_z"] = 0.001 # config["sigma_z_vals"][0]
if "rotationAngles" in config:
    config["rotationAngle"] = 0 #config["rotationAngles"][0]
arg_list = (rzArray, config)


# --- Create the radDist
if config["distType"] == "Helical":
    rD = Util_radDist.radDist_Helical_parallel(arg_list, return_result=True)
elif config["distType"] == "HelicalRing":
    rD = Util_radDist.radDist_HelicalRing_parallel(arg_list, return_result=True)
elif config["distType"] == "ElongatedRing":
    rD = Util_radDist.radDist_ElongatedRing_parallel(arg_list, return_result=True)
elif config["distType"] == "SquareTube":
    rD = Util_radDist.radDist_SquareTube_parallel(arg_list, return_result=True)
else:
    raise RuntimeError(
        "Please have 'elongatedRing', 'helical', 'HelicalRing', or 'SqureTube' in the configFileName"
    )


# --- Plot everything ---
if rD is not None:
    fig = rD.plotOverview(plot_etendue=[""], return_figure=True)
    # Add the pellet scatter
    if fig is not None:
        pellet_scatter(fig.axes[2])
        plt.show()
