# emis3d/run_config.py
"""Loads and verifys the run configuration file loaded"""

from dataclasses import dataclass, field, fields
from emis3d.globals import SCALE_DEFINITIONS, ERROR_DEFINITIONS, FIT_METHODS


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
    # --- Channel errors (Util_emis3D.channel_error). floor_relative:
    # sigma^2 = (error_floor * peak)^2 + (error_relative * signal)^2, with the
    # floor raised per channel to the noise measured in noise_window, if given
    error_definition: str = "floor_relative"
    error_floor: float = 0.05  # fraction of the peak signal
    error_relative: float = 0.05  # fraction of each channel's signal
    noise_window: list | None = None  # [t_start, t_end], a quiet period before the event
    fit_method: str = "least_squares"  # or "differential_evolution" (global, slower)
    enable_dphi_scaling: bool = True
    vary_peak_rad_location: bool = False
    peak_shift_max_degrees: float = 180.0  # How far the injector can stray from the starting location
    helical_endpoint_weight: float = 1.0
    numProcessorsFitting: int = 1
    paths: dict = field(default_factory=dict)
    # Ranges for TPF and other derived quantities (emis3d.uncertainty.UncertaintyConfig)
    uncertainty: dict = field(default_factory=dict)

    def __post_init__(self):
        # Check for valid scale definitions
        if self.scale_def not in SCALE_DEFINITIONS:
            raise ValueError(f"Unknown scale_def: {self.scale_def!r}")
        # Check for valid error definitions
        # (case-insensitive, like Util_emis3D.signal_error)
        if self.error_definition.lower() not in ERROR_DEFINITIONS:
            hint = {
                "inverse": "use 'constant': it has the same shape, and the overall error scale changes neither the fit nor its ranking",
                "inverse square": "it is now called 'poisson'",
            }.get(self.error_definition.lower(), f"choose from {ERROR_DEFINITIONS}")
            raise ValueError(f"Unknown error_definition {self.error_definition!r}: {hint}")
        if self.fit_method not in FIT_METHODS:
            raise ValueError(f"Unknown fit_method: {self.fit_method!r}; choose from {FIT_METHODS}")
        if self.error_floor < 0 or self.error_relative < 0:
            raise ValueError("error_floor and error_relative must be >= 0")
        if self.error_definition.lower() == "floor_relative" and self.error_floor == 0 and self.noise_window is None:
            raise ValueError("floor_relative needs error_floor > 0 or a noise_window, or near-zero channels get zero error")
        if self.noise_window is not None and (len(self.noise_window) != 2 or self.noise_window[0] >= self.noise_window[1]):
            raise ValueError("noise_window must be [t_start, t_end] with t_start < t_end")
        for name, bolo in self.BOLOMETERS.items():
            bad = (bolo or {}).get("badChannels", [])
            if not isinstance(bad, list):
                raise ValueError(f"BOLOMETERS.{name}.badChannels must be a list of channel names")
        # --- Uncertainty block: known options, calibration for configured cameras only
        from emis3d.uncertainty import UncertaintyConfig

        unc = UncertaintyConfig.from_info({"uncertainty": self.uncertainty})
        unknown_cams = set(unc.calibration) - set(self.BOLOMETERS)
        if unknown_cams:
            raise ValueError(f"uncertainty.calibration lists cameras not in BOLOMETERS: {sorted(unknown_cams)}")
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
