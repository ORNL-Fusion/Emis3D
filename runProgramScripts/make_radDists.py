# make_radDists.py
"""
This script will generate Helical radDists for a given shot and gfile

This was done since the GUI is too cryptic

Only run this file using the command line, example:
python make_radDists.py

Written by Jeffrey Herfindal, Feb. 13, 2025

Overall radDist creation procedure used in this program:
1. Create a config file under /inputs/{tokamakName}/{shot}/
2. This program loads that configuration file
3. This program creates the R, z grid
4. This program then loops over the input sigma_R and sigma_z values,
   creating many radDists within the input/{tokamakName}/radDists/{folder name from config}


Major changes
1: radDists now utilize many processors, creating more than one at once which
can greatly speed up the process
2. Configuration files are used for the: tokamak, radDist, bolometers.
3. Program is now less case specific, the program utilizes dictionaries for radDist,
fitting routines, etc. which can easiliy be looped over and don't require specific
definitions for each radDist type (e.g. self.first_punc, self.second_punc, etc.)
4. General clean up of the code. Definitions utilized by multiple classes were moved
to Util and Util_radDist files
5. Removed outdated packages, replaced them with their current version (e.g. scipy.interp2d)
6.



TODO:


"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from concurrent.futures import ProcessPoolExecutor

import main.Util_radDist as Util_radDist
from main.Globals import *
from main.Tokamak import Tokamak
from main.Util import config_loader
import numpy as np

if __name__ == "__main__":

    # ------------------------------------------------------
    # --- Only change these variables! ---
    # ------------------------------------------------------
    # This should be the path after /inputs/{tokamak}/radDists/:
    tokamakName = "DIII-D"
    configFileName = "helical_config.yaml"  # "elongatedRing_config.yaml"
    tokamakName = "JET"
    configFileName = "elongatedRing_config.yaml"  # "elongatedRing_config.yaml"
    # ------------------------------------------------------
    # --- Do not change anything below this line ---
    # ------------------------------------------------------

    # --- Start of program
    pathFileName = os.path.join(
        EMIS3D_INPUTS_DIRECTORY, tokamakName, "radDists", configFileName
    )

    # --- Load the configuration file, if it exists
    if os.path.isfile(pathFileName):
        config = config_loader(pathFileName)

    else:
        config = None
        print(
            f"Could not load the configuration file, file does not exist: {pathFileName}"
        )

    if config is not None:
        print(f"Loaded configuration file: {pathFileName}")
    else:
        print("Exiting program")
        sys.exit()

    numProcessors = int(config["numProcessors"])

    # --- Need to load the tokamak to get wall information, this tokamak class will not be passed to the radDists
    tok = Tokamak(
        tokamakName=tokamakName,
        mode="Analysis",
        reflections=False,
        eqFileName=config["eqFileName"],
    )

    rzArray = None

    # --- Load the tokamak if rArray and zArray are blank in the configuration file
    if len(config["GRID"]["rLimits"]) == 0 or len(config["GRID"]["zLimits"]) == 0:
        rzArray = Util_radDist.callRZGridTokamak(
            tok,
            num_r=config["GRID"]["NumRStartGrid"],
            num_z=config["GRID"]["NumZStartGrid"],
        )

    else:
        if tok.wall is not None:

            rzArray = Util_radDist.createRZGrid(
                r_limits=config["GRID"]["rLimits"],
                z_limits=config["GRID"]["zLimits"],
                num_r=config["GRID"]["NumRStartGrid"],
                num_z=config["GRID"]["NumZStartGrid"],
                wallcurve=tok.wall["wallcurve"],
            )

    # --- Remove sigma_R and sigma_z_vals from the config file since we don't
    # need to pass all of them to each radDist
    sigma_R_vals = config["sigma_R_vals"].copy()
    sigma_z_vals = config["sigma_z_vals"].copy()
    rotationAngles = config["rotationAngles"].copy()
    del config["sigma_R_vals"], config["sigma_z_vals"], config["rotationAngles"]

    if rzArray is not None:
        # --- Build the full task list up front: one (rz, config) task per
        # radDist, with a config snapshot per parameter combination.
        arg_list = []
        for rotationAngle in rotationAngles:
            for sigma_R in sigma_R_vals:
                for sigma_z in sigma_z_vals:

                    # --- Skip rotation angles for circles (sigma_z == sigma_R):
                    # only the rotationAngle = 0 case is unique
                    if sigma_z == sigma_R and rotationAngle > 0.0:
                        continue
                    # --- Skip duplicates: rotationAngle 90 with sigma_z equal
                    # to one of the sigma_R values reproduces a 0-degree case
                    if rotationAngle == 90 and sigma_z in sigma_R_vals:
                        continue

                    cfg = dict(config)
                    cfg["sigma_R"] = sigma_R
                    cfg["sigma_z"] = sigma_z
                    cfg["rotationAngle"] = rotationAngle
                    arg_list.extend((val, cfg) for val in rzArray)

        # CHANGE THIS BELOW, let's go back to the older
        worker_fn = {
            "Helical": Util_radDist.radDist_Helical_parallel,
            "ElongatedRing": Util_radDist.radDist_ElongatedRing_parallel,
        }.get(config["distType"])
        if worker_fn is None:
            print(f"Unknown distType: {config['distType']}")
            sys.exit()

        print(f"Submitting {len(arg_list)} radDists to {numProcessors} workers")

        # --- ONE executor for the whole run. Each worker builds the tokamak
        # once (initializer) and reuses it for every radDist it processes;
        # observations inside workers run on a SerialEngine, so this is the
        # only layer of parallelism.
        with ProcessPoolExecutor(
            max_workers=numProcessors,
            initializer=Util_radDist.init_worker_tokamak,
            initargs=(tokamakName, config["eqFileName"]),
        ) as executor:
            # --- Explicitly consume results so the iterator is cleared and
            # worker exceptions surface
            for _ in executor.map(worker_fn, arg_list):
                pass

    else:
        print(f"Problem making the rzArray")

"""
# OLD SCRIPT: which went below for sigma_z in sigma_z_vals:


# --- Split the rzArray to conserve memory during the process pool executor,
# try to have it split up evenly between the numbe of processors used
num_split = 1
if rzArray.shape[0] > (numProcessors - 1.0):
    num_split = np.floor((rzArray.shape[0] / (numProcessors - 1.0)))
rzArray_split = np.array_split(rzArray, num_split)
for rz in rzArray_split:
    # --- Skip rotation angle if the sigma_R and sigma_z are equal (aka a circle)
    # only do the case where the rotationAngle = 0
    if sigma_z == sigma_R and rotationAngle > 0.0:
        pass
    # --- Skip where sigma_z is a sigma_R value for rotationAngle 90
    if rotationAngle == 90 and sigma_z in sigma_R_vals:
        pass
    else:
        # --- Add stuff to the config, create list of r, z points to solve at
        # this sigma_z and sigma_R
        config["sigma_R"] = sigma_R
        config["sigma_z"] = sigma_z
        config["rotationAngle"] = rotationAngle
        arg_list = [(val, config) for val in rz]

        # --- Start computation in parrallel minus 2 of your processors (so you can actually use your computer)
        if config["distType"] == "Helical":
            with ProcessPoolExecutor(
                max_workers=numProcessors
            ) as executor:
                # --- Explicitly consume results so iterator is cleared
                for _ in executor.map(
                    Util_radDist.radDist_Helical_parallel, arg_list
                ):
                    pass
        elif config["distType"] == "ElongatedRing":
            with ProcessPoolExecutor(
                max_workers=numProcessors
            ) as executor:
                # --- Explicitly consume results so iterator is cleared
                for _ in executor.map(
                    Util_radDist.radDist_ElongatedRing_parallel,
                    arg_list,
                ):
                    pass
"""
