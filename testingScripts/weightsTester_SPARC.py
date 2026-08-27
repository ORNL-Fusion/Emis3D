"""
This program will plot radiated power as calculated by a weighted average
for a NIMROD simulation

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
from matplotlib import colormaps
from os.path import join
import json

tokamakName = "SPARC_FDR"

startTimestep = 60
stopTimestep = 2820
interval = 60
radDist_dir_path = join(EMIS3D_INPUTS_DIRECTORY, tokamakName, "radDists", "Test", "NIMROD", "Nonresonant")
times = np.arange(startTimestep, stopTimestep+interval, interval)


ptots = []
brightnesses = []
# collect radiated powers and channel brightnesses from each timestep
for time in times:
    timestep = f"{time:05d}"
    radDist_path = join(radDist_dir_path, f"R_{time:.2f}_z_{time:.2f}.json")

    with open(radDist_path) as file:
        radDist_properties = json.load(file)

    prad_tot = radDist_properties["data"]["toroidalRadiatedPower"]["NIMROD"]["P_total"]
    ptots.append(prad_tot)

    brightnesses_tstep = radDist_properties["data"]["Brightness"]["NIMROD"]
    brightnesses_tstep_vec = [brightnesses_tstep[key] for key in brightnesses_tstep.keys()]
    brightnesses.append(brightnesses_tstep_vec)
ptots = np.array(ptots)

# make weights based on a given timestep
ref_timestep = 2760#1260#2520#2580#2640#2700#2760 #2820
ref_timestep_indx = np.argwhere(times == ref_timestep).squeeze()
ref_brightnesses = np.array(brightnesses[ref_timestep_indx]).squeeze()
ref_Prad = ptots[ref_timestep_indx]
brightnessSquareSum = np.sum(ref_brightnesses**2)
weights = []
for channelIndx in range(len(ref_brightnesses)):
    channelBrightness = ref_brightnesses[channelIndx]
    weight = channelBrightness * ref_Prad / brightnessSquareSum
    weights.append(weight)
weights = np.array(weights)


# Calculate weighted sum Prads
weighted_prads = []
for timeIndx in range(len(times)):
    brightnesses_tstep = np.array(brightnesses[timeIndx]).squeeze()
    weightedPrad = np.sum(brightnesses_tstep * weights)
    weighted_prads.append(weightedPrad)
    #breakpoint()
weighted_prads = np.array(weighted_prads)

fontsize=14
ticksize=12
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[6.0, 3.0])

fieldnames = ["NIMROD", "Weighted"]
varnames = ["NIMROD", "Weighted"]
labels = ["NIMROD", "Weighted"]
linestyles = ['-', '--', '-.', ':']
linewidths = [3, 3, 3, 3]
colors = colormaps["tab10"]
plotdata = [ptots*1e-9, weighted_prads*1e-9]

for indx in range(len(fieldnames)):
    axs.plot(times * 1e-3, plotdata[indx],\
        linestyle=linestyles[indx], linewidth=linewidths[indx],\
        color = colors(indx),\
        label=labels[indx])

axs.set_ylabel("Prad (GW)", fontsize=fontsize)
axs.set_xlabel('Time (ms)', fontsize=fontsize)
axs.tick_params(axis='x', labelsize=ticksize)
axs.tick_params(axis='y', labelsize=ticksize) 
axs.set_title(f"Reference Time (ms): {ref_timestep*1e-3:.2f}")
axs.legend(loc="upper left", fontsize=fontsize)
plt.tight_layout()
#plt.savefig(f'IP_{shotnumber}.png', bbox_inches='tight', pad_inches=0.05, dpi=600)
plt.show()