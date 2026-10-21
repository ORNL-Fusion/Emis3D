# emis3d/run_config.py
"""
Run configuration: loads and validates the YAML file that describes one
Emis3D analysis (which data, which radDists, how to fit them).

Every key in the file must be a field below; unknown keys are an error, so a
typo cannot silently fall back to a default. Angles are in degrees, times in
the units of the measurement files. File paths are relative to the
workspace locations shown by `emis3d paths`.
"""

from dataclasses import dataclass, field, fields
from emis3d.globals import SCALE_DEFINITIONS, ERROR_DEFINITIONS, FIT_METHODS


@dataclass
class RunConfig:
    # --- What is being analysed
    BOLOMETERS: dict  # cameras to fit: name -> {dataFileName, scalingFactor (default 1), badChannels (default [])}
    shot: int  # shot number; also used in result file names and path templates
    tokamakName: str | None = None  # machine, e.g. "JET"; selects machines/, measurements/, raddists/, equilibria/

    # --- Candidate radiation distributions, folders relative to <raddists>/<tokamak>
    radDistDirectories_LocIndependent: list = field(default_factory=list)  # e.g. elongated rings
    radDistDirectories_LocDependent: list = field(default_factory=list)  # e.g. helical; combined per injector
    injectionLocation: float = 0.0  # injector angle [deg]; must match the radDists' (the fit uses theirs; plots use this)

    # --- Times
    tstart: float | None = None  # start of the analysis window (for reference; times are passed to _perform_fits)
    tend: float | None = None  # end of the analysis window (for reference)
    Nt: int | None = None  # number of times in the window (for reference)
    dt: float | None = None  # averaging half-width: data are averaged over [t - dt, t + dt]

    # --- Toroidal shape of the radiation (how emission falls off away from the injector)
    scale_def: str = "linear"  # "linear", "exponential", "constant" or "step"
    enable_dphi_scaling: bool = True  # fit the fall-off (b); False keeps b fixed and fits only the amplitude
    vary_peak_rad_location: bool = False  # True fits the peak's toroidal location instead of fixing it at the injector
    peak_shift_max_degrees: float = 180.0  # how far the fitted peak may move from the injector [deg]
    helical_endpoint_weight: float = 1.0  # ties clockwise/counterclockwise helical endpoints; 0 = off; helical only

    # --- Channel errors (util_emis3D.channel_error). floor_relative:
    # sigma^2 = (error_floor * peak)^2 + (error_relative * signal)^2, with the
    # floor raised per channel to the noise measured in noise_window, if given
    error_definition: str = "floor_relative"  # or "constant", "poisson", "exponential"
    error_floor: float = 0.05  # additive error, as a fraction of the peak signal
    error_relative: float = 0.05  # relative error, as a fraction of each channel's signal
    noise_window: list | None = None  # [t_start, t_end], a quiet period before the event

    # --- Fitting
    fit_method: str = "least_squares"  # or "differential_evolution" (global, slower; for a fitted peak location)
    numProcessorsFitting: int = 1  # parallel fits in scripts/run_emis3D.py

    # --- Optional blocks
    paths: dict = field(default_factory=dict)  # override workspace locations for this run (see emis3d.workspace)
    uncertainty: dict = field(default_factory=dict)  # TPF and other ranges (see emis3d.uncertainty.UncertaintyConfig)

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
