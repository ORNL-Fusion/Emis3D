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

Equivalent command: emis3d make-raddists <config> [--tokamak NAME] [--workers N] [--dry-run]
"""

import argparse

from emis3d.pipeline import make_raddists

if __name__ == "__main__":  # required: worker processes re-import this file on macOS
    # --- Defaults, used when no arguments are given
    tokamakName = "JET"
    configFileName = "elongatedRing_config.yaml"  # a path, or a name in the workspace's raddists/<tokamak> folder

    parser = argparse.ArgumentParser(description="Generate a radDist library")
    parser.add_argument("--tokamak", default=tokamakName)
    parser.add_argument("--config", default=configFileName, help="radDist config: a path, or a name in the 'raddists' location")
    parser.add_argument("--workspace", default=None, help="workspace folder (default: nearest emis3d.yaml, EMIS3D_RUNS_DIR or user config)")
    parser.add_argument("--workers", type=int, default=None, help="radDists built in parallel (default: the config's numProcessors)")
    parser.add_argument("--dry-run", action="store_true", help="only count what would be built")
    args = parser.parse_args()

    summary = make_raddists(args.config, tokamak=args.tokamak, workspace=args.workspace, workers=args.workers, dry_run=args.dry_run)
    verb = "Would build" if args.dry_run else "Built"
    print(f"{verb} {summary['radDists']} {summary['distType']} radDists "
          f"({summary['shapes']} shape(s) x {summary['start_positions']} start positions) in {summary['folder']}")
