# emis3d/run_config.py
"""Loads and verifys the run configuration file loaded"""

from dataclasses import dataclass, field, fields
from emis3d.Globals import SCALE_DEFINITIONS, ERROR_DEFINITIONS


@dataclass
class RunConfig:
    BOLOMETERS: dict
    shot: int
    tokamakName: str | None = None
    radDistDirectories_LocIndependent: list = field(default_factory=list)
    radDistDirectories_LocDependent: list = field(default_factory=list)
    injectionLocation: float = 0.0
    tstart: float | None = None
    tend: float | None = None
    Nt: int | None = None
    dt: float | None = None
    scale_def: str = "linear"
    error_definition: str = "exponential"
    enable_dphi_scaling: bool = True
    vary_peak_rad_location: bool = False
    crossCalib: bool = False
    helical_endpoint_weight: float = 1.0
    numProcessorsFitting: int = 1
    paths: dict = field(default_factory=dict)

    def __post_init__(self):
        # Check for valid scale definitions
        if self.scale_def not in SCALE_DEFINITIONS:
            raise ValueError(f"Unknown scale_def: {self.scale_def!r}")
        # Check for valid error definitions
        # (case-insensitive, like Util_emis3D.signal_error)
        if self.error_definition.lower() not in ERROR_DEFINITIONS:
            raise ValueError(f"Unknown error_definition: {self.error_definition!r}")
        # Check if the radDist directories are listed
        if not (self.radDistDirectories_LocDependent or self.radDistDirectories_LocIndependent):
            raise ValueError("No radDist directories configured")

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown run-config keys: {sorted(unknown)}")
        return cls(**d)
