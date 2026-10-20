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


# Availble toroidal scaling functions
SCALE_DEFINITIONS: list[str] = ["linear", "exponential", "constant", "step"]

# Availble error definitions for the data
ERROR_DEFINITIONS: list[str] = ["exponential", "inverse", "inverse square", "constant"]

# Explicit export list
__all__ = ["SUPPORTED_TOKAMAKS", "SCALE_DEFINITIONS", "ERROR_DEFINITIONS"]
