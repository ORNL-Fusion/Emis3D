# make_radDists.py
"""
This script will generate Helical radDists for a given shot and gfile

This was done since the GUI is too cryptic

Only run this file using the command line, examples:
python make_radDists.py
python make_radDists.py --tokamak DIII-D --config helical_config.yaml
python make_radDists.py --config /path/to/helical_config.yaml --workspace ~/projects/emis3d-runs

Written by Jeffrey Herfindal, Feb. 13, 2025

Overall radDist creation procedure used in this program:
1. Create a config file in the workspace's 'raddists' location (or give its path)
2. This program loads that configuration file
3. This program creates the R, z grid
4. This program then loops over the input sigma_R and sigma_z values,
   creating many radDists within <raddists>/{saveRunsDirectoryName from config}
   (see src/emis3d/workspace.py for where 'raddists' points)


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

import argparse
import os
import sys


from concurrent.futures import ProcessPoolExecutor

import emis3d.Util_radDist as Util_radDist
from emis3d.Globals import *
from emis3d.Tokamak import Tokamak
from emis3d.Util import config_loader
from emis3d.workspace import Workspace, set_workspace

if __name__ == "__main__":

    # ------------------------------------------------------
    # --- Only change these variables! ---
    # ------------------------------------------------------
    # configFileName is a path, or a name inside the workspace's 'raddists' location
    # (or pass --tokamak and --config, e.g. --tokamak DIII-D --config helical_config.yaml)
    tokamakName = "JET"
    configFileName = "elongatedRing_config.yaml"
    # ------------------------------------------------------
    # --- Do not change anything below this line ---
    # ------------------------------------------------------

    # --- Command-line arguments override the defaults above
    parser = argparse.ArgumentParser(description="Generate a radDist library")
    parser.add_argument("--tokamak", default=tokamakName)
    parser.add_argument("--config", default=configFileName,
                        help="radDist config: a path, or a name in the 'raddists' location")
    parser.add_argument("--workspace", default=None,
                        help="workspace folder (default: nearest emis3d.yaml, EMIS3D_RUNS_DIR or user config)")
    args = parser.parse_args()
    tokamakName = args.tokamak

    # --- Start of program
    ws = Workspace.resolve(workspace=args.workspace)
    set_workspace(ws)
    ws.export_env()  # worker processes resolve the same workspace
    if os.path.isfile(os.path.expanduser(args.config)):
        pathFileName = os.path.expanduser(args.config)
    else:
        pathFileName = str(ws.path("raddists", tokamak=tokamakName) / args.config)

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
        # Only distinct shapes: circles, rotated twins and parameters the
        # radDist type ignores would otherwise rebuild the same radDist
        shapes = Util_radDist.shape_parameter_sets(
            config["distType"], sigma_R_vals, sigma_z_vals, rotationAngles
        )
        print(f"{len(shapes)} distinct shape(s) x {len(rzArray)} start positions")
        arg_list = []
        for params in shapes:
            cfg = dict(config)
            cfg.update(params)
            arg_list.extend((val, cfg) for val in rzArray)

        # CHANGE THIS BELOW, let's go back to the older
        worker_fn = {
            "Helical": Util_radDist.radDist_Helical_parallel,
            "ElongatedRing": Util_radDist.radDist_ElongatedRing_parallel,
            "HelicalRing": Util_radDist.radDist_HelicalRing_parallel,
            "SquareTube": Util_radDist.radDist_SquareTube_parallel,
        }.get(config["distType"])
        if worker_fn is None:
            print(f"Unknown distType: {config['distType']}")
            sys.exit()

        print(f"Submitting {len(arg_list)} radDists to {numProcessors} workers")

        with ProcessPoolExecutor(
            max_workers=numProcessors,
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
