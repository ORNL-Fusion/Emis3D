# Constants.py
"""
Package-wide path constants and configuration

Override the data root at runtime via the EMIS3D_ROOT environment variable:
    export EMIS3D_ROOT=/path/to/your/emis3d_data

"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Root of the user data tree. Override with EMIS3D_ROOT to decouple data
# from the source checkout (required when installed via pip)
EMIS3D_PARENT_DIRECTORY: Path = Path(os.environ.get("EMIS3D_ROOT", _REPO_ROOT))

# Tokamak configuration file locations
EMIS3D_TOKMAK_DIRECTORY: Path = EMIS3D_PARENT_DIRECTORY / "tokamaks"

# User-created input data (equilibria, radDists, run configs, etc.)
# This directory is not committed to the repository
EMIS3D_INPUTS_DIRECTORY: Path = EMIS3D_PARENT_DIRECTORY / "inputs"


SUPPORTED_TOKAMAKS: list[str] = ["DIII-D", "SPARC", "JET", "KSTAR"]


# Explicit export list
__all__ = [
    "EMIS3D_PARENT_DIRECTORY",
    "EMIS3D_TOKMAK_DIRECTORY",
    "EMIS3D_INPUTS_DIRECTORY",
    "SUPPORTED_TOKAMAKS",
]
