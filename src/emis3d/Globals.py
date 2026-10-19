# Constants.py
"""
Package-wide constants.

File locations are not constants: emis3d.workspace resolves where machines,
equilibria, measurements, radDists, runs, results and cache live. Use
``emis3d.workspace.get_workspace()``.
"""

from __future__ import annotations

import numpy as np

# Machines with a known toroidal-angle convention. Any machine whose settings
# file is found on the workspace machine path can be loaded; see emis3d.workspace.
SUPPORTED_TOKAMAKS: list[str] = ["MOCK", "DIII-D", "SPARC", "SPARC_FDR", "JET"]
# Angle conventions used in each tokamak are different from that used in Cherab.
# Emis3D uses the Cherab angle convention. This angle is subtracted in the evaluate
# statements in RadDist to make the angles match.
TOR_CONVENTION_PHIS = {
    "MOCK": 0.0,
    "JET": np.pi / 2.0,
    "SPARC": 0.0,
    "SPARC_FDR": 0.0,
    "DIII-D": 0.0,
}

# Availble toroidal scaling functions
SCALE_DEFINITIONS: list[str] = ["linear", "exponential", "constant", "step"]

# Availble error definitions for the data
ERROR_DEFINITIONS: list[str] = ["exponential", "inverse", "inverse square", "constant"]

# Explicit export list
__all__ = ["SUPPORTED_TOKAMAKS", "TOR_CONVENTION_PHIS", "SCALE_DEFINITIONS", "ERROR_DEFINITIONS"]
