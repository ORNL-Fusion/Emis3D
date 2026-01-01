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
4. This program then loops over the input polSigma and elongation values,
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

sys.path.append(os.path.abspath("../.."))

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
    configFileName = "184407_injectionLocation_225/helical_config.yaml"
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

    # --- Load the tokamak if rArray and zArray are blank in the configuration file
    if len(config["GRID"]["rLimits"]) == 0 or len(config["GRID"]["zLimits"]) == 0:
        # --- Need to load the tokamak to get wall information
        tok = Tokamak(
            tokamakName=tokamakName,
            mode="Analysis",
            reflections=False,
            eqFileName=config["eqFileName"],
        )
        rzArray = Util_radDist.callRZGridTokamak(
            tok,
            numRgrid=config["GRID"]["NumRStartGrid"],
            numZgrid=config["GRID"]["NumZStartGrid"],
        )
    else:
        rzArray = Util_radDist.createRZGrid(
            rLimits=config["GRID"]["rLimits"],
            zLimits=config["GRID"]["zLimits"],
            numRgrid=config["GRID"]["NumRStartGrid"],
            numZgrid=config["GRID"]["NumZStartGrid"],
            wallcurve=None,
        )

    # --- Remove polSigma and elongations from the config file since we don't
    # need to pass all of them to each radDist
    polSigmas = config["polSigmas"].copy()
    elongations = config["elongations"].copy()
    rotationAngles = config["rotationAngles"].copy()
    del config["polSigmas"], config["elongations"], config["rotationAngles"]

    for rotationAngle in rotationAngles:
        for elongation in elongations:
            for polSigma in polSigmas:
                # --- Split the rzArray to conserve memory during the process pool executor,
                # try to have it split up evenly between the numbe of processors used
                num_split = np.floor((rzArray.shape[0] / (numProcessors - 1.0)))
                rzArray_split = np.array_split(rzArray, num_split)
                for rz in rzArray_split:
                    # --- Skip rotation angle if the elongation and polSigma are equal (aka a circle)
                    # only do the case where the rotationAngle = 0
                    if elongation == polSigma and rotationAngle > 0.0:
                        pass
                    else:
                        # --- Add stuff to the config, create list of r, z points to solve at
                        # this elongation and polsigma
                        config["polSigma"] = polSigma
                        config["elongation"] = elongation
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
