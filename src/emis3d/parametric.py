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
SCALE_DEFS = ("constant", "exponential", "linear", "step")


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
    amplitude: dict = field(default_factory=dict)  # absolute amplitude(s); the fitted a values are relative to the data's peak
    toroidal: dict = field(default_factory=dict)  # step scale: {"locations_deg": [...], "amplitudes": [...] (relative)}

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
    scale_def  : "constant", "exponential", "linear" or "step". The step gives every
                 toroidal sector the amplitude of its nearest camera location (the
                 cameras' toroidal angles): one amplitude per location, a_<deg>,
                 all solved exactly at every trial
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
            raise ValueError(f"the parametric fit supports scale_def {', '.join(SCALE_DEFS)}, not {scale_def!r}")
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

        # --- Amplitudes: one, or one per camera location for the step scale
        self.step_deg: list[float] = []
        self._regions = {}
        if scale_def == "step":
            G0 = self.matrices[0]
            rows0 = self._rows[id(G0)][0]
            in_fit = {G0.cameras[i] for i in np.unique(G0.camera_of_row[rows0])}
            angles = {b.name: float(b.info["CAMERA_POSITION_R_Z_PHI"][2]) % 360.0 for b in template.tokamak.bolometers if b.name in in_fit}
            missing_cams = sorted(in_fit - set(angles))
            if missing_cams:
                raise ValueError(f"step scale: no camera position for {missing_cams} in {template.info['tokamakName']}'s camera configs")
            self.step_deg = sorted({round(a, 6) for a in angles.values()})
            locs = np.deg2rad(self.step_deg)
            for G in self.matrices:  # each sector belongs to its nearest camera location, around the torus
                gap = np.abs((G.grid.phi_centers[:, None] - locs[None, :] + np.pi) % (2.0 * np.pi) - np.pi)
                self._regions[id(G)] = np.argmin(gap, axis=1)
        self.amp_names = [f"a_{int(round(d))}" for d in self.step_deg] if self.step_deg else ["a"]
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

    def design(self, G, values: dict) -> np.ndarray:
        """
        The model's signals per amplitude, on matrix G, in data order: one
        column (the toroidal shape applied) or, for the step scale, one column
        per camera location (the signal from the sectors nearest it).
        """
        self._apply(values)
        rows, _, _ = self._rows[id(G)]
        eps = G.emissivity_on_grid(self.template, self.emission, supersample=self.supersample)
        P = G.signals_by_sector(eps)[rows]
        self.evaluations += 1
        if self.step_deg:
            region = self._regions[id(G)]
            return np.column_stack([P[:, region == j].sum(axis=1) for j in range(len(self.step_deg))])
        return (P @ self._toroidal(G, values.get("b", 0.0)))[:, None]

    def model_signals(self, G, values: dict) -> np.ndarray:
        """The channels' signals with every amplitude 1, on matrix G, in data order."""
        return self.design(G, values).sum(axis=1)

    def _scale_ref(self, d: np.ndarray, m: np.ndarray) -> float:
        """Absolute amplitude for a = 1: the model's peak scaled to the data's peak."""
        peak_m = float(np.max(np.abs(m))) if m.size else 0.0
        peak_d = float(np.max(np.abs(d))) if d.size else 0.0
        return peak_d / peak_m if peak_m > 0 and peak_d > 0 else 1.0

    def _amplitudes(self, X: np.ndarray, d: np.ndarray, sig: np.ndarray) -> tuple[np.ndarray, float]:
        """The amplitudes (relative) solved exactly within their bounds, and the scale to absolute."""
        from scipy.optimize import lsq_linear

        ref = self._scale_ref(d, X.sum(axis=1))
        A = (X * ref) / sig[:, None]
        if X.shape[1] == 1:  # one amplitude: the bounded optimum is the clipped one
            denom = float(A[:, 0] @ A[:, 0])
            a = float(A[:, 0] @ (d / sig)) / denom if denom > 0 else 0.0
            return np.array([np.clip(a, self.a_bounds[0], self.a_bounds[1])]), ref
        res = lsq_linear(A, d / sig, bounds=(self.a_bounds[0], self.a_bounds[1]))
        return np.asarray(res.x, dtype=float), ref

    def chi2(self, G, values: dict) -> tuple[float, np.ndarray | float]:
        """
        (chi2, a): the amplitude(s) solved exactly within their bounds (a is
        relative to the data's peak); a is a float for one amplitude, an array
        (one per camera location) for the step scale.
        """
        _, d, sig = self._rows[id(G)]
        X = self.design(G, values)
        a, ref = self._amplitudes(X, d, sig)
        chi2 = float(np.sum(((d - ref * (X @ a)) / sig) ** 2))
        return chi2, (float(a[0]) if a.size == 1 else a)

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
        for name, a0 in zip(self.amp_names, np.atleast_1d(a_start), strict=True):
            pars.add(name, value=float(np.clip(a0, self.a_bounds[0], self.a_bounds[1])), min=self.a_bounds[0], max=self.a_bounds[1])

        def residual(p):
            vals = {n: p[n].value for n in self.names}
            X = self.design(G1, vals)
            a = np.array([p[n].value for n in self.amp_names])
            return (d - self._scale_ref(d, X.sum(axis=1)) * (X @ a)) / sig

        fit = minimize(residual, pars, method="least_squares")
        fitted = cast(Any, fit).params
        values = {n: float(fitted[n].value) for n in list(self.names) + self.amp_names}
        stderr = {n: (float(fitted[n].stderr) if fitted[n].stderr is not None else None) for n in values}
        chi2_1 = float(fit.chisqr)
        stages.append({"matrix": G1.grid.describe(), "evaluations": self.evaluations, "chi2": chi2_1, "seconds": round(time.perf_counter() - t0, 1)})
        say(f"  polish ({G1.grid.describe()}): chi2/N {chi2_1 / max(self.n_data, 1):.2f} after {self.evaluations} evaluations, "
            f"{time.perf_counter() - t0:.0f} s")
        flags = at_bounds(values, {**self.bounds, **{n: self.a_bounds for n in self.amp_names}})
        for flag in flags:
            logger.warning("%s fit: %s (widen the bound, or consider another model)", self.model, flag)
        X = self.design(G1, {n: values[n] for n in self.names})
        ref = self._scale_ref(d, X.sum(axis=1))
        amplitude = {n: values[n] * ref for n in self.amp_names}
        toroidal = {"locations_deg": list(self.step_deg), "amplitudes": [values[n] for n in self.amp_names]} if self.step_deg else {}
        self._apply(values)
        return ParametricResult(model=self.model, values=values, stderr=stderr, chi2=chi2_1, n_data=len(d),
                                aic=float(fit.aic), stages=stages, fit=fit, n_params=len(self.names) + len(self.amp_names),
                                at_bounds=flags, amplitude=amplitude, toroidal=toroidal)
