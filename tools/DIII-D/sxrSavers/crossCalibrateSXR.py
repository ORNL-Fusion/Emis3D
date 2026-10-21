# crossCalibrateSXR.py
"""
Cross-calibration factors between the DIII-D SXR arrays.

The radiation is assumed to be axisymmetric during the current quench, so
every array should see the same axisymmetric radDist. emis3d.calibration fits
each array's amplitude against every radDist in the config's
radDistDirectories_LocIndependent (elongated rings, for example), picks the
best radDist at each time, and reports each array's correction factor relative
to a reference array.

Order of operations:
1) grabSXRData.py to grab and save the SXR data
2) this program (or `emis3d calibrate`) to find the cross-calibration factors
3) put the printed BOLOMETERS block (scalingFactor values) in your run configs

Equivalent command:
    emis3d calibrate SXR_crossCalib_Config.yaml --times 2124

NOTE: You will need pre-generated elongated ring radDists for this to work.

# Written by Jeffrey Herfindal, Aug. 26, 2025; moved onto emis3d.calibration
"""

import os

from emis3d.calibration import cross_calibrate

# --- Values to change ---
path_to_config = os.path.join(os.path.dirname(__file__), "SXR_crossCalib_Config.yaml")
times = [2124.0]          # one or more times during the current quench
reference = None          # e.g. "DISRADU"; default: the array with the largest amplitude

# --- Start of program ---
result = cross_calibrate(path_to_config, times, reference=reference)
print(result.table())
print()
print(result.yaml_block())
