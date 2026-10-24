# Constants.py
"""
Package-wide constants.

File locations are not constants: emis3d.workspace resolves where machines,
equilibria, measurements, radDists, runs, results and cache live. Use
``emis3d.workspace.get_workspace()``.
"""

from __future__ import annotations

# Machines with a known toroidal-angle convention. Any machine whose settings
# file is found on the workspace machine path can be loaded; see emis3d.workspace.
SUPPORTED_TOKAMAKS: list[str] = ["MOCK", "DIII-D", "SPARC", "SPARC_FDR", "JET"]


# Availble toroidal scaling functions
SCALE_DEFINITIONS: list[str] = ["linear", "exponential", "constant", "step"]

# Availble error definitions for the data
ERROR_DEFINITIONS: list[str] = ["floor_relative", "constant", "poisson", "exponential"]

# Default bounds of the fit's amplitude a (relative to the data's peak: 1 means the
# model's peak equals the data's) and toroidal decay b (per scale_def); the run
# config's `bounds: {a: [lo, hi], b: [lo, hi]}` overrides them, for both solvers
AMPLITUDE_BOUNDS: tuple[float, float] = (0.05, 20.0)
DECAY_BOUNDS: dict[str, tuple[float, float]] = {"exponential": (0.0, 10.0), "linear": (0.01, 0.4)}

# How the fit is solved (run config `solver`): a radDist library, or a model's
# parameters through geometry matrices
SOLVERS: list[str] = ["library", "parametric"]

# Minimizers a run config can choose (fit_method)
FIT_METHODS: list[str] = ["least_squares", "differential_evolution"]

# Explicit export list
# Version of the saved radDist format. RadDistFitting refuses any other
# version: radDists saved by earlier Emis3D versions must be regenerated.
RADDIST_FORMAT: int = 2

__all__ = ["SUPPORTED_TOKAMAKS", "SCALE_DEFINITIONS", "ERROR_DEFINITIONS", "FIT_METHODS", "RADDIST_FORMAT", "AMPLITUDE_BOUNDS",
           "DECAY_BOUNDS", "SOLVERS"]
