# radDist_dataTester_JET.py
"""
Loads the data for a particular time slice, used to figoure out a decent radDist

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
from emis3d.Emis3D import Emis3D

def pellet_scatter(ax):
    """Gives SPI trajectory range, for plotting"""

    y_in = [0.75, 1.88]
    x_in = [2.23, 3.17]

    y_out = [0.406, 1.88]
    x_out = [2.91, 3.17]

    ax.plot(x_in, y_in, linestyle = 'dashed', color = 'red', linewidth = 2.0)
    ax.plot(x_out, y_out, linestyle = 'dashed', color = 'red', linewidth = 2.0)


evalTime = 50.954

tokamakName = "JET"
runConfigName = "95709/95709_runConfig.yaml"
configFileName = "elongatedRing_config.yaml"  
verbose = True
rzvalues = [3.2, 1.3]


# --- Load the data
t = Emis3D(
    tokamakName=tokamakName,
    runConfigName=runConfigName,
    verbose=verbose,
    initialize=True,
)
t._prepare_fits(evalTime=evalTime, crossCalib=False)





# --- Create the radDist using only one point, we don't need to loop over everything
pathFileName = get_workspace().path("raddists", tokamak=tokamakName) / configFileName
config = config_loader(pathFileName)
print(pathFileName)
if config is None:
    raise FileNotFoundError(f"Could not load config file: {pathFileName}")


# --- Update the configuration file
rzArray = np.array([rzvalues[0], rzvalues[1]])

if "sigma_R_vals" in config:
    config["sigma_R"] = config["sigma_R_vals"][0]
if "sigma_z_vals" in config:
    config["sigma_z"] = config["sigma_z_vals"][0]
if "rotationAngles" in config:
    config["rotationAngle"] = config["rotationAngles"][0]
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
    
    if fig is not None:
        # Add the pellet scatter
        pellet_scatter(fig.axes[2])
        

        # Add the bolometer data
        for ii in range(3, 5):
            bolo = fig.axes[ii].get_title()
            ax = fig.axes[ii]
            channels = t._channel_numbers(bolo)
            data_ = t.fitData[evalTime]['boloData'][bolo]
            # Normalize the data 
            data_ /= np.nanmax(data_)
            ax.scatter(channels, data_, marker = 's', color = 'black')

        plt.show()