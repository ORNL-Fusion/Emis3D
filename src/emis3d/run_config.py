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
from emis3d.globals import SCALE_DEFINITIONS, ERROR_DEFINITIONS, FIT_METHODS, SOLVERS, AMPLITUDE_BOUNDS, DECAY_BOUNDS


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
    # --- How the fit is solved: "library" fits every radDist in the radDistDirectories;
    # "parametric" fits each model's parameters through geometry matrices (the parametric block)
    solver: str = "library"
    # Bounds of the amplitude a (relative to the data's peak) and the toroidal decay b, both
    # solvers: {a: [lo, hi], b: [lo, hi]}; defaults in emis3d.globals (AMPLITUDE_BOUNDS, DECAY_BOUNDS)
    bounds: dict = field(default_factory=dict)
    # The parametric solver (emis3d.parametric): {models: [ring, shell, tube], matrices: [low, medium],
    #  equilibrium: <g-file>, bounds: {model: {param: [lo, hi]}}, fixed: {model: {param: value}},
    #  maxiter, popsize, seed, supersample}
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
        # --- The solver, and what it needs
        if self.solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {self.solver!r}")
        self._check_bounds()
        if self.solver == "parametric":
            self._check_parametric()
        elif not (self.radDistDirectories_LocDependent or self.radDistDirectories_LocIndependent):
            raise ValueError("solver: library needs radDistDirectories_LocIndependent or _LocDependent (or use solver: parametric)")

    def _check_bounds(self) -> None:
        unknown = sorted(set(self.bounds) - {"a", "b"})
        if unknown:
            raise ValueError(f"bounds takes a and b, not {unknown} (model parameters go under parametric.bounds)")
        for k, v in self.bounds.items():
            if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) and v[0] < v[1]):
                raise ValueError(f"bounds.{k} must be [low, high] with low < high, got {v!r}")
        if "a" in self.bounds and self.bounds["a"][0] < 0:
            raise ValueError("bounds.a must not go below 0")

    def amplitude_bounds(self) -> tuple[float, float]:
        """Bounds of the amplitude a: the config's, else AMPLITUDE_BOUNDS."""
        return tuple(self.bounds.get("a", AMPLITUDE_BOUNDS))  # type: ignore[return-value]

    def decay_bounds(self) -> tuple[float, float]:
        """Bounds of the toroidal decay b: the config's, else DECAY_BOUNDS for the scale_def."""
        default = DECAY_BOUNDS.get(self.scale_def.lower(), DECAY_BOUNDS["exponential"])
        return tuple(self.bounds.get("b", default))  # type: ignore[return-value]

    def _check_parametric(self) -> None:
        from emis3d.parametric import MODELS

        p = self.parametric
        if not p:
            raise ValueError("solver: parametric needs a parametric block (models, matrices, ...)")
        allowed = {"models", "matrices", "equilibrium", "bounds", "fixed", "maxiter", "popsize", "seed", "supersample"}
        unknown = sorted(set(p) - allowed)
        if unknown:
            hint = " (use models: [ring], a list)" if "model" in unknown else ""
            raise ValueError(f"Unknown parametric options {unknown}{hint}; the block takes {sorted(allowed)}")
        models = p.get("models")
        if not isinstance(models, list) or not models:
            raise ValueError(f"parametric.models must be a list of models to fit and compare, from {sorted(MODELS)}")
        bad = sorted(set(models) - set(MODELS))
        if bad:
            raise ValueError(f"parametric.models: unknown model(s) {bad}; choose from {sorted(MODELS)}")
        matrices = p.get("matrices") or ["medium"]
        if not isinstance(matrices, list) or not all(isinstance(m, str) for m in matrices):
            raise ValueError("parametric.matrices must be a list of matrix names (coarsest first), e.g. [low, medium]")
        for key in ("bounds", "fixed"):
            block = p.get(key) or {}
            stray = sorted(set(block) - set(MODELS))
            if stray:
                raise ValueError(f"parametric.{key} is per model ({sorted(MODELS)}): {{ring: {{sigma_R: ...}}}}; "
                                 f"{stray} are not models (a and b go under the top-level bounds)")
            for model, params in block.items():
                names = set(MODELS[model]["params"])
                extra = sorted(set(params or {}) - names)
                if extra:
                    raise ValueError(f"parametric.{key}.{model} names {extra}, which the {model} model does not have ({sorted(names)})")
                if key == "bounds":
                    for name, v in (params or {}).items():
                        if not (isinstance(v, (list, tuple)) and len(v) == 2 and v[0] < v[1]):
                            raise ValueError(f"parametric.bounds.{model}.{name} must be [low, high] with low < high, got {v!r}")
        if "shell" in models and not p.get("equilibrium"):
            raise ValueError("parametric.equilibrium (a g-file name) is required for the shell model")
        if self.scale_def.lower() not in ("constant", "exponential", "linear"):
            raise ValueError("the parametric solver supports scale_def constant, exponential or linear")

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
