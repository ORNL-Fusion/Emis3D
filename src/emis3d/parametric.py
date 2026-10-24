# parametric.py
"""
Fitting a radiation model's parameters directly, through the geometry
matrix (emis3d.geometry), with no radDist library.

The model is a radDist type with free shape parameters (a ring's position,
widths and tilt; a flux-surface shell's psi and width; ...) plus the
toroidal shape (scale_def with its decay b). For each trial the emissivity
is evaluated on the voxels the cameras see, scaled toroidally at every
sector's own angle, multiplied by G and compared with the data; the
amplitude is solved in closed form at every trial (variable projection), so
the search is only over the shape parameters.

The search is global then local: differential evolution on the coarsest
matrix, then a least-squares polish on the finest (coarse-to-fine), which
also gives the parameter uncertainties.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from lmfit import Parameters, minimize
from scipy.optimize import differential_evolution

import emis3d.util_emis3D as Util_emis3D

logger = logging.getLogger(__name__)

# Model name -> radDist type and its free parameters with default bounds (None:
# from the vessel's extent, set in ParametricFit). The run config's
# parametric.bounds.<model> overrides any of them. A ring's tilt covers 0-90
# degrees: with both widths free, that is every orientation once (sigma_R,
# sigma_z, tilt) and (sigma_z, sigma_R, tilt + 90) are the same ellipse.
MODELS = {
    "ring": {"distType": "ElongatedRing", "emission": "elongatedRing",
             "params": {"startR": None, "startZ": None, "sigma_R": (0.03, 1.5), "sigma_z": (0.03, 1.5), "rotationAngle": (0.0, 90.0)}},
    "tube": {"distType": "SquareTube", "emission": "squareTube",
             "params": {"startR": None, "startZ": None, "dR": (0.02, 1.0), "dz": (0.02, 1.5)}},
    "shell": {"distType": "FluxShell", "emission": "fluxShell",
              "params": {"psi_center": (0.1, 1.15), "psi_width": (0.02, 0.5)}},
}
SCALE_DEFS = ("constant", "exponential", "linear")


def at_bounds(values: dict, bounds: dict, rel_tol: float = 1e-3) -> list[str]:
    """
    Parameters whose fitted value sits on a bound (within rel_tol of the
    range): e.g. ["sigma_R at its upper bound 1.5"]. A fit pushing against a
    bound usually wants a shape the model cannot take: widen the bound, or
    try another model.
    """
    flags = []
    for name, (lo, hi) in bounds.items():
        if name not in values:
            continue
        tol = rel_tol * (hi - lo)
        if values[name] - lo <= tol:
            flags.append(f"{name} at its lower bound {lo:g}")
        elif hi - values[name] <= tol:
            flags.append(f"{name} at its upper bound {hi:g}")
    return flags


@dataclass
class ParametricResult:
    model: str
    values: dict  # every fitted value: shape parameters, b (if any), a
    stderr: dict  # 1-sigma uncertainties from the final stage (None where unavailable)
    chi2: float
    n_data: int
    aic: float
    stages: list = field(default_factory=list)  # one record per stage: matrix, evaluations, chi2, seconds
    fit: object = None  # lmfit result of the final stage
    n_params: int = 0  # fitted parameters, the amplitude included
    at_bounds: list = field(default_factory=list)  # e.g. ["b at its lower bound 0"]
    amplitude: float = float("nan")  # the absolute amplitude (a is relative to the data's peak)

    def summary(self) -> str:
        parts = [f"{k} {v:.4g}" + (f" +- {self.stderr[k]:.2g}" if self.stderr.get(k) else "") for k, v in self.values.items()]
        text = f"{self.model}: " + ", ".join(parts) + f" | chi2/N {self.chi2 / max(self.n_data, 1):.2f}"
        return text + (f" | AT BOUNDS: {'; '.join(self.at_bounds)}" if self.at_bounds else "")


def comparison_table(results: list) -> str:
    """The fitted models, best (lowest AIC) first: chi2/N, AIC, delta AIC, parameters, flags."""
    ranked = sorted(results, key=lambda r: r.aic)
    best = ranked[0].aic if ranked else 0.0
    lines = [f"{'model':7s} {'chi2/N':>7s} {'AIC':>9s} {'dAIC':>7s} {'k':>3s}  parameters"]
    for r in ranked:
        params = ", ".join(f"{k} {v:.3g}" for k, v in r.values.items())
        lines.append(f"{r.model:7s} {r.chi2 / max(r.n_data, 1):7.2f} {r.aic:9.1f} {r.aic - best:7.1f} {r.n_params:3d}  {params}")
        if r.at_bounds:
            lines.append(f"{'':7s} {'':>7s} {'':>9s} {'':>7s} {'':>3s}  ! at bounds: {'; '.join(r.at_bounds)}")
    return "\n".join(lines)


class ParametricFit:
    """
    model      : a key of MODELS
    template   : a built radDist of that type (its settings are changed per trial)
    matrices   : GeometryMatrix objects, coarsest first (one is fine)
    data       : {channel tag: (value, sigma)} in the matrix's units; sigma inf excludes a channel
    scale_def  : "constant", "exponential" or "linear"
    bounds     : {parameter: (low, high)} for the model's parameters, overriding MODELS
    fixed      : {parameter: value}
    a_bounds   : bounds of the amplitude, relative to the data's peak (1: the model's
                 peak equals the data's); default emis3d.globals.AMPLITUDE_BOUNDS
    b_bounds   : bounds of the toroidal decay; default DECAY_BOUNDS for the scale_def
    """

    def __init__(self, model: str, template, matrices: list, data: dict, scale_def: str, injection_deg: float,
                 bounds: dict | None = None, fixed: dict | None = None, supersample=None,
                 a_bounds: tuple | None = None, b_bounds: tuple | None = None):
        from emis3d.globals import AMPLITUDE_BOUNDS, DECAY_BOUNDS

        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; choose from {sorted(MODELS)}")
        if scale_def not in SCALE_DEFS:
            raise ValueError(f"the parametric fit supports scale_def constant, exponential or linear, not {scale_def!r}")
        self.model, self.spec, self.template = model, MODELS[model], template
        self.matrices = list(matrices)
        self.scale_def, self.mu = scale_def, float(np.deg2rad(injection_deg))
        self.emission = self.spec["emission"]
        self.supersample = supersample
        self.fixed = dict(fixed or {})

        wall = template.tokamak.wall
        defaults = {"startR": (wall["minr"] + 0.05, wall["maxr"] - 0.05), "startZ": (wall["minz"] + 0.05, wall["maxz"] - 0.05)}
        self.bounds = {}
        for name, b in self.spec["params"].items():
            if name in self.fixed:
                continue
            self.bounds[name] = tuple((bounds or {}).get(name) or b or defaults[name])
        self.has_decay = scale_def in DECAY_BOUNDS
        if self.has_decay:
            self.bounds["b"] = tuple(b_bounds if b_bounds is not None else DECAY_BOUNDS[scale_def])
        self.a_bounds = tuple(a_bounds if a_bounds is not None else AMPLITUDE_BOUNDS)
        self.names = list(self.bounds)  # search order

        # --- Data aligned with each matrix's rows
        self.data = data
        self._rows = {}
        for G in self.matrices:
            tags = {t: i for i, t in enumerate(G.channels)}
            keep = [t for t in data if np.isfinite(data[t][1]) and data[t][1] > 0]  # bad channels (infinite error) are left out
            missing = sorted(set(keep) - set(tags))
            if missing:
                raise ValueError(f"channels not in the geometry matrix ({G.grid.describe()}): {missing}; "
                                 "drop them from the data (Emis3D does this) or rebuild the matrix with their cameras")
            self._rows[id(G)] = (np.array([tags[t] for t in keep]), np.array([data[t][0] for t in keep], dtype=float),
                                 np.array([data[t][1] for t in keep], dtype=float))
        self.n_data = len(self._rows[id(self.matrices[0])][0])
        self._f_cache = {}
        self.evaluations = 0

    # ------------------------------------------------------------------
    def _apply(self, values: dict) -> None:
        for k, v in self.fixed.items():
            self.template.info[k] = float(v)
        for k, v in values.items():
            if k != "b":
                self.template.info[k] = float(v)

    def _toroidal(self, G, b: float) -> np.ndarray:
        """The toroidal factor at each sector's angle (a = 1)."""
        if not self.has_decay:
            return np.ones(G.grid.n_phi)
        return np.asarray(Util_emis3D.scale_wrapper(a=1.0, b=float(b), phi=G.grid.phi_centers, mu=self.mu, scale_def=self.scale_def,
                                                     emissionName=self.emission), dtype=float)

    def model_signals(self, G, values: dict) -> np.ndarray:
        """The channels' signals (a = 1) for the parameters, on matrix G, in data order."""
        self._apply(values)
        rows, _, _ = self._rows[id(G)]
        eps = G.emissivity_on_grid(self.template, self.emission, supersample=self.supersample)
        P = G.signals_by_sector(eps)[rows]
        self.evaluations += 1
        return P @ self._toroidal(G, values.get("b", 0.0))

    def _scale_ref(self, d: np.ndarray, m: np.ndarray) -> float:
        """Absolute amplitude for a = 1: the model's peak scaled to the data's peak."""
        peak_m = float(np.max(np.abs(m))) if m.size else 0.0
        peak_d = float(np.max(np.abs(d))) if d.size else 0.0
        return peak_d / peak_m if peak_m > 0 and peak_d > 0 else 1.0

    def chi2(self, G, values: dict) -> tuple[float, float]:
        """
        (chi2, a): the amplitude solved in closed form, then held within its
        bounds (a is relative to the data's peak; for a one-parameter
        quadratic, clipping gives the bounded optimum exactly).
        """
        _, d, sig = self._rows[id(G)]
        m = self.model_signals(G, values)
        w = 1.0 / sig**2
        ref = self._scale_ref(d, m)
        denom = float(np.sum(w * m * m))
        a_abs = float(np.sum(w * d * m)) / denom if denom > 0 else 0.0
        a = float(np.clip(a_abs / ref, self.a_bounds[0], self.a_bounds[1]))
        return float(np.sum(w * (d - a * ref * m) ** 2)), a

    # ------------------------------------------------------------------
    def search(self, seed: int = 0, maxiter: int = 60, popsize: int = 12, tol: float = 1e-3, progress=None) -> ParametricResult:
        """Global search on the coarsest matrix, then a least-squares polish on the finest."""
        say = progress or (lambda text: None)
        stages = []
        lo = np.array([self.bounds[n][0] for n in self.names])
        hi = np.array([self.bounds[n][1] for n in self.names])

        # --- Stage 1: differential evolution on the coarsest matrix
        G0 = self.matrices[0]
        t0 = time.perf_counter()
        self.evaluations = 0
        objective = lambda x: self.chi2(G0, dict(zip(self.names, x, strict=True)))[0]  # noqa: E731
        de = differential_evolution(objective, list(zip(lo, hi, strict=True)), maxiter=maxiter, popsize=popsize, tol=tol,
                                    polish=False, **cast(Any, {"seed": seed}))  # the stub does not list seed
        best = dict(zip(self.names, de.x, strict=True))
        chi2_0, a0 = self.chi2(G0, best)
        stages.append({"matrix": G0.grid.describe(), "evaluations": self.evaluations, "chi2": chi2_0, "seconds": round(time.perf_counter() - t0, 1)})
        say(f"  global search ({G0.grid.describe()}): chi2/N {chi2_0 / max(self.n_data, 1):.2f} after {self.evaluations} evaluations, "
            f"{time.perf_counter() - t0:.0f} s")

        # --- Stage 2: least squares on the finest matrix, amplitude free, for the covariance
        G1 = self.matrices[-1]
        t0 = time.perf_counter()
        self.evaluations = 0
        rows, d, sig = self._rows[id(G1)]
        pars = Parameters()
        for n in self.names:
            pars.add(n, value=float(np.clip(best[n], lo[self.names.index(n)], hi[self.names.index(n)])), min=self.bounds[n][0], max=self.bounds[n][1])
        _, a_start = self.chi2(G1, best)
        pars.add("a", value=float(np.clip(a_start, self.a_bounds[0], self.a_bounds[1])), min=self.a_bounds[0], max=self.a_bounds[1])

        def residual(p):
            vals = {n: p[n].value for n in self.names}
            m = self.model_signals(G1, vals)
            return (d - p["a"].value * self._scale_ref(d, m) * m) / sig

        fit = minimize(residual, pars, method="least_squares")
        fitted = cast(Any, fit).params
        values = {n: float(fitted[n].value) for n in list(self.names) + ["a"]}
        stderr = {n: (float(fitted[n].stderr) if fitted[n].stderr is not None else None) for n in values}
        chi2_1 = float(fit.chisqr)
        stages.append({"matrix": G1.grid.describe(), "evaluations": self.evaluations, "chi2": chi2_1, "seconds": round(time.perf_counter() - t0, 1)})
        say(f"  polish ({G1.grid.describe()}): chi2/N {chi2_1 / max(self.n_data, 1):.2f} after {self.evaluations} evaluations, "
            f"{time.perf_counter() - t0:.0f} s")
        flags = at_bounds(values, {**self.bounds, "a": self.a_bounds})
        for flag in flags:
            logger.warning("%s fit: %s (widen the bound, or consider another model)", self.model, flag)
        amplitude = values["a"] * self._scale_ref(d, self.model_signals(G1, {n: values[n] for n in self.names}))
        self._apply(values)
        return ParametricResult(model=self.model, values=values, stderr=stderr, chi2=chi2_1, n_data=len(d),
                                aic=float(fit.aic), stages=stages, fit=fit, n_params=len(self.names) + 1,
                                at_bounds=flags, amplitude=float(amplitude))
