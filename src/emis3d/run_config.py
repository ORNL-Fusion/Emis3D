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

import numpy as np
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

    # --- Times: Nt evenly spaced evaluation times from tstart to tend, inclusive
    # (see eval_times; tstart alone, or tend == tstart, gives one time)
    tstart: float | None = None  # first evaluation time, in the units of the measurement files
    tend: float | None = None  # last evaluation time (default: tstart)
    Nt: int | None = None  # number of evaluation times from tstart to tend (default: 1)
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
    # Fit a model's parameters through geometry matrices instead of a radDist library (emis3d.parametric):
    # {model: ring|tube|shell, matrices: [low, medium], equilibrium: <g-file>, bounds: {param: [lo, hi]},
    #  fixed: {param: value}, maxiter, popsize, seed, supersample}
    parametric: dict = field(default_factory=dict)

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
        # --- Evaluation times
        if self.tstart is None and (self.tend is not None or self.Nt is not None):
            raise ValueError("tend and Nt need tstart")
        if self.tstart is not None and self.tend is not None and self.tend < self.tstart:
            raise ValueError(f"tend ({self.tend}) is before tstart ({self.tstart})")
        if self.Nt is not None and (int(self.Nt) != self.Nt or self.Nt < 1):
            raise ValueError(f"Nt must be a whole number >= 1, got {self.Nt}")
        # --- Parametric fitting needs no radDist library; otherwise directories must be listed
        if self.parametric:
            from emis3d.parametric import MODELS

            allowed = {"model", "matrices", "equilibrium", "bounds", "fixed", "maxiter", "popsize", "seed", "supersample"}
            unknown = sorted(set(self.parametric) - allowed)
            if unknown:
                raise ValueError(f"Unknown parametric options {unknown}; the block takes {sorted(allowed)}")
            model = self.parametric.get("model")
            if model not in MODELS:
                raise ValueError(f"parametric.model must be one of {sorted(MODELS)}, got {model!r}")
            matrices = self.parametric.get("matrices") or ["medium"]
            if not isinstance(matrices, list) or not all(isinstance(m, str) for m in matrices):
                raise ValueError("parametric.matrices must be a list of matrix names (coarsest first), e.g. [low, medium]")
            params = set(MODELS[model]["params"]) | {"b"}
            for key in ("bounds", "fixed"):
                bad = sorted(set(self.parametric.get(key) or {}) - params)
                if bad:
                    raise ValueError(f"parametric.{key} names parameters {bad} that the {model} model does not have ({sorted(params)})")
            if model == "shell" and not self.parametric.get("equilibrium"):
                raise ValueError("parametric.equilibrium (a g-file name) is required for the shell model")
            if self.scale_def.lower() not in ("constant", "exponential", "linear"):
                raise ValueError("the parametric fit supports scale_def constant, exponential or linear")
        elif not (self.radDistDirectories_LocDependent or self.radDistDirectories_LocIndependent):
            raise ValueError("No radDist directories configured (or a parametric block)")

    def eval_times(self) -> list[float]:
        """
        Evaluation times from tstart, tend and Nt: Nt evenly spaced times from
        tstart to tend, inclusive. tstart alone, or tend == tstart, gives one
        time (whatever Nt is). Times are rounded to 1e-9 so they can be used as
        dictionary keys (run.bestFits[50.951]). Empty if tstart is not set.
        """
        if self.tstart is None:
            return []
        tend = self.tstart if self.tend is None else self.tend
        if tend == self.tstart:
            return [float(self.tstart)]
        n = int(self.Nt) if self.Nt is not None else 1
        return [float(t) for t in np.round(np.linspace(self.tstart, tend, n), 9)]

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown run-config keys: {sorted(unknown)}")
        return cls(**d)
