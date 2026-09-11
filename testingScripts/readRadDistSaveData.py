# readRadDistSaveData.py
"""
Reads radDists under a given directory, saves them as an h5 file, mocking
how the signal would be during a shot.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_REPO_ROOT))

import numpy as np
from main.Globals import EMIS3D_INPUTS_DIRECTORY
from os.path import join
import json
from main.Util import save_json

tokamakName = "SPARC_FDR"

# --- Stores to home/username/Documents
SAVE_DIR = Path.home() / "Documents"

FILENAME = "SPARC_FDR_NIMROD.h5"

startTimestep = 60
stopTimestep = 2820
interval = 60
radDist_dir_path = join(
    EMIS3D_INPUTS_DIRECTORY,
    tokamakName,
    "radDists",
    "Test",
    "NIMROD",
    "NIMROD_sigma_R_0.1_sigma_z_0.001",
)
times = np.arange(startTimestep, stopTimestep + interval, interval)

data = {}


def make_bolo_dicts(rD):
    """
    Determines bolometer names, number of channels in each bolometer,
    and names of each channel. Creates a dict for each bolometer, to be
    filled in with data. This is currently specific to how SPARC's
    bolometer config files are organized
    """

    bolos = {}

    chOrder = rD["data"]["Brightness"]["channelOrder"]

    # --- First determine the bolometer names
    for key in chOrder:
        parts = key.split("_")
        bolo = f"{parts[0]}_{parts[1]}"
        if bolo not in bolos:
            bolos[bolo] = {}

        bolos[bolo][key] = []

    return bolos


# collect radiated powers and channel brightnesses from each timestep
for ii, time in enumerate(times):
    timestep = f"{time:05d}"
    radDist_path = join(radDist_dir_path, f"R_{time:.2f}_z_{time:.2f}.json")

    with open(radDist_path) as file:
        radDist_properties = json.load(file)

    # --- Create the empty dict
    if ii == 0:
        data = make_bolo_dicts(radDist_properties)
        data["time"] = []
        data["prad"] = []

    prad_tot = radDist_properties["data"]["toroidalRadiatedPower"]["NIMROD"]["P_total"]
    data["prad"].append(float(prad_tot))
    data["time"].append(float(time))

    brightnesses_tstep = radDist_properties["data"]["Brightness"]["NIMROD"]

    for key in brightnesses_tstep:
        parts = key.split("_")
        bolo_ = f"{parts[0]}_{parts[1]}"
        data[bolo_][key].append(float(np.sum(brightnesses_tstep[key])))

save_json(data, SAVE_DIR, FILENAME)
print(f"File saved to: {SAVE_DIR}/{FILENAME}")
