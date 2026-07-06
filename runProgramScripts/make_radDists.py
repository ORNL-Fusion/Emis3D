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
import time


def validate_config(config):
    """Validate radDist config values that otherwise fail inside worker processes."""
    required_keys = [
        "distType",
        "tokamakName",
        "saveRunsDirectoryName",
        "injectionLocation",
        "eqFileName",
        "sigma_R_vals",
        "sigma_z_vals",
        "rotationAngles",
        "numProcessors",
        "GRID",
        "BOLOMETER_PROPS",
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Config is missing required key(s): {missing_keys}")

    if config["distType"] == "Helical":
        helical_keys = ["sigmaKernel", "numFieldLines"]
        missing_keys = [key for key in helical_keys if key not in config]
        if missing_keys:
            raise KeyError(
                "Helical configs are missing required key(s): "
                f"{missing_keys}. Add sigmaKernel and numFieldLines."
            )

        sigma_kernel = float(config["sigmaKernel"])
        invalid_sigmas = [
            sigma_R
            for sigma_R in config["sigma_R_vals"]
            if sigma_kernel >= float(sigma_R)
        ]
        if invalid_sigmas:
            raise ValueError(
                "Helical sigmaKernel must be smaller than every sigma_R target. "
                f"sigmaKernel={sigma_kernel}, invalid sigma_R_vals={invalid_sigmas}"
            )


if __name__ == "__main__":

    t_start = time.perf_counter()

    # ------------------------------------------------------
    # --- Only change these variables! ---
    # ------------------------------------------------------
    # This should be the path after /inputs/{tokamak}/radDists/:
    tokamakName = "KSTAR"
    configFileName = "helical_config_KSTAR_26525.yaml" # "helical_config.yaml"  # "elongatedRing_config.yaml"
    # tokamakName = "JET"
    # configFileName = "elongatedRing_config.yaml"  # "elongatedRing_config.yaml"
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
        validate_config(config)
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

    if rzArray is not None and rzArray.shape[0] == 0:
        print(
            "No R/z grid points were inside the wall. Increase "
            "GRID.NumRStartGrid/GRID.NumZStartGrid or set GRID.rLimits/zLimits."
        )
        rzArray = None

    # --- Remove sigma_R and sigma_z_vals from the config file since we don't
    # need to pass all of them to each radDist
    sigma_R_vals = config["sigma_R_vals"].copy()
    sigma_z_vals = config["sigma_z_vals"].copy()
    rotationAngles = config["rotationAngles"].copy()
    del config["sigma_R_vals"], config["sigma_z_vals"], config["rotationAngles"]

    parameter_sets = []
    seen_parameter_sets = set()
    for sigma_R in sigma_R_vals:
        for sigma_z in sigma_z_vals:
            # Circular rings are independent of rotation angle, so build them once.
            angles = [0.0] if sigma_z == sigma_R else rotationAngles
            for rotationAngle in angles:
                key = (sigma_R, sigma_z, rotationAngle)
                if key not in seen_parameter_sets:
                    parameter_sets.append(key)
                    seen_parameter_sets.add(key)

    if rzArray is not None:
        print(
            f"Building {len(parameter_sets)} parameter set(s) "
            f"at {rzArray.shape[0]} R/z point(s)."
        )
        for sigma_R, sigma_z, rotationAngle in parameter_sets:
            # --- Split the rzArray to conserve memory during the process pool executor,
            # try to have it split up evenly between the numbe of processors used
            num_split = 1
            if rzArray.shape[0] > (numProcessors - 1.0):
                num_split = int(
                    np.floor(rzArray.shape[0] / max(numProcessors - 1, 1))
                )
            rzArray_split = np.array_split(rzArray, num_split)
            for rz in rzArray_split:
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
    else:
        print(f"Problem making the rzArray")

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    print(f"\nmake_radDists.py completed in {elapsed:.2f} seconds")
    print(f"Equivalent: {elapsed/60:.2f} minutes")
