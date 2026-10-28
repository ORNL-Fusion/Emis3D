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
    # A shell brighter on one poloidal side (a crescent): asym_angle is the poloidal angle of the
    # bright side about the magnetic axis, degrees (0 outboard midplane, 90 top, 180 inboard)
    "asymshell": {"distType": "FluxShell", "emission": "fluxShell",
                  "params": {"psi_center": (0.1, 1.15), "psi_width": (0.02, 0.5), "asym_amp": (0.0, 1.0), "asym_angle": (-180.0, 180.0)}},
    # A helical structure: a Gaussian source on the injector's plane carried along field lines
    # (emis3d.fieldline_map) with a decay exp(-decay dphi^2); decay takes bounds.b. The toroidal
    # shape is the decay itself, so no sector scale is applied (the step scale still can be)
    "helical": {"distType": "FieldAligned", "emission": "fieldAligned", "field_aligned": True, "source": "rz",
                "params": {"startR": None, "startZ": None, "sigma_R": (0.02, 0.5), "sigma_z": (0.02, 0.5)}},
    # The same structure from a source in flux coordinates: thin across flux surfaces (psi_width), extended
    # along one (theta_width, degrees; up to a full ring), centred at theta_center (0 outboard, 90 top)
    "helical_psi": {"distType": "FieldAligned", "emission": "fieldAligned", "field_aligned": True, "source": "psi",
                    "params": {"psi_center": (0.1, 1.15), "psi_width": (0.01, 0.4), "theta_center": (-180.0, 180.0), "theta_width": (3.0, 180.0)}},
    # A spot on the SPI spray (the run config's injector): in the injector's plane (its source plane is
    # fixed there), spray_s along the spray from the shatter point and spray_offset_deg from the centre
    # line; the bounds default to the vessel and the cone. Carried along the field like the helical model
    "spray_spot": {"distType": "FieldAligned", "emission": "fieldAligned", "field_aligned": True, "source": "spray", "fixed_plane": True,
                   "params": {"spray_s": None, "spray_offset_deg": None, "sigma_R": (0.02, 0.3), "sigma_z": (0.02, 0.3)}},
}
NEEDS_INJECTOR = ("spray_spot",)


def default_bounds(template) -> dict:
    """Bounds a model takes from its template: the vessel for positions, the spray for a spray spot."""
    wall = template.tokamak.wall
    out = {"startR": (wall["minr"] + 0.05, wall["maxr"] - 0.05), "startZ": (wall["minz"] + 0.05, wall["maxz"] - 0.05)}
    if template.info.get("source") == "spray":
        half = float(template.info["injector"]["divergence_deg"]) / 2.0
        out["spray_s"] = (0.0, float(template.info["spray_length"]))
        out["spray_offset_deg"] = (-half, half)
    return out
# The helical model's source plane: with vary_peak_rad_location the toroidal angle phi_source [deg]
# is fitted within peak_shift_max_degrees of the injector (the whole structure rotates with it)
NEEDS_EQUILIBRIUM = ("shell", "asymshell", "helical", "helical_psi", "spray_spot")
PERIODIC_PARAMS = ("rotationAngle", "theta_center")  # their range's ends are not limits


def is_periodic(name: str) -> bool:
    """A periodic parameter, by its own name or a composite's prefixed one (band_theta_center)."""
    return any(name == p or name.endswith("_" + p) for p in PERIODIC_PARAMS)
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
    components: list = field(default_factory=list)  # composites: per component {name, model, values, amplitudes}

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
                 a_bounds: tuple | None = None, b_bounds: tuple | None = None,
                 vary_peak: bool = False, peak_shift_max_degrees: float = 180.0, toroidal_floor: bool = False):
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

        defaults = default_bounds(template)
        self.bounds = {}
        for name, b in self.spec["params"].items():
            if name in self.fixed:
                continue
            self.bounds[name] = tuple((bounds or {}).get(name) or b or defaults[name])
        self.field_aligned = bool(self.spec.get("field_aligned"))
        self.injection_deg = float(injection_deg)
        if self.field_aligned:  # the decay along the field lines is the model's own toroidal shape
            self.bounds["decay"] = tuple(b_bounds if b_bounds is not None else DECAY_BOUNDS["exponential"])
            if "phi_source" in self.fixed:
                pass
            elif vary_peak and not self.spec.get("fixed_plane"):  # the source plane's toroidal angle, around the injector
                self.bounds["phi_source"] = tuple((bounds or {}).get("phi_source")
                                                  or (self.injection_deg - float(peak_shift_max_degrees), self.injection_deg + float(peak_shift_max_degrees)))
            else:
                self.fixed["phi_source"] = self.injection_deg
        self.has_decay = scale_def in DECAY_BOUNDS and not self.field_aligned
        if self.has_decay:
            self.bounds["b"] = tuple(b_bounds if b_bounds is not None else DECAY_BOUNDS[scale_def])
            # with vary_peak the peaked profile's centre is fitted with the shape (as a field-aligned source plane is)
            if vary_peak and "phi_peak" not in self.fixed:
                inj = float(injection_deg)
                self.bounds["phi_peak"] = tuple((bounds or {}).get("phi_peak")
                                                or (inj - float(peak_shift_max_degrees), inj + float(peak_shift_max_degrees)))
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
        # a toroidally uniform floor under the peak: one more amplitude, solved exactly with a
        self.floor = bool(toroidal_floor) and self.has_decay and not self.step_deg
        if self.floor:
            self.amp_names.append("c")
        self._f_cache = {}
        self.evaluations = 0

    # ------------------------------------------------------------------
    def _apply(self, values: dict) -> None:
        for k, v in self.fixed.items():
            self.template.info[k] = float(v)
        for k, v in values.items():
            if k != "b":
                self.template.info[k] = float(v)

    def _toroidal(self, G, b: float, values: dict | None = None) -> np.ndarray:
        """The toroidal factor at each sector's angle (a = 1), peaked at phi_peak when it is fitted, else the injector."""
        if not self.has_decay:
            return np.ones(G.grid.n_phi)
        mu = float(np.deg2rad(values["phi_peak"])) if values is not None and "phi_peak" in values else self.mu
        return np.asarray(Util_emis3D.scale_wrapper(a=1.0, b=float(b), phi=G.grid.phi_centers, mu=mu, scale_def=self.scale_def,
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
        peaked = P @ self._toroidal(G, values.get("b", 0.0), values)
        if self.floor:
            return np.column_stack([peaked, P.sum(axis=1)])  # the peak, and the uniform floor
        return peaked[:, None]

    def voxel_columns(self, G, values: dict, only_seen: bool = True) -> list:
        """
        The model's emissivity on G's voxels, one field per amplitude column
        (each with its toroidal shape applied): design(G, values)[:, j] equals
        G.signals(column j) at the fitted channels. The fit, the composite
        radDist and the ranges all use these fields. only_seen evaluates only
        the voxels a camera sees (enough for signals); power needs them all.
        """
        self._apply(values)
        eps = G.emissivity_on_grid(self.template, self.emission, supersample=self.supersample, only_seen=only_seen)
        sector = G.sector_of_voxel()
        if self.step_deg:
            region = self._regions[id(G)][sector]
            return [eps * (region == j) for j in range(len(self.step_deg))]
        cols = [eps * self._toroidal(G, values.get("b", 0.0), values)[sector]]
        return cols + [eps] if self.floor else cols

    def voxel_field(self, G, values: dict, only_seen: bool = True) -> np.ndarray:
        """
        The fitted emissivity on G's voxels, amplitudes included (absolute).
        only_seen=False evaluates every voxel, as radiated power needs (the
        signals, and so the amplitudes, are the same either way).
        """
        cols = self.voxel_columns(G, values, only_seen=only_seen)
        rows, d, _ = self._rows[id(G)]
        X = np.column_stack([G.signals(c)[rows] for c in cols])
        ref = self._scale_ref(d, X.sum(axis=1))
        out = np.zeros_like(np.asarray(cols[0], dtype=float))
        for name, col in zip(self.amp_names, cols, strict=True):
            out += float(values[name]) * ref * col
        return out

    def sector_profile(self, G, values: dict) -> np.ndarray:
        """Radiated power in each toroidal sector of G's grid (sum over the poloidal plane of emissivity x voxel volume)."""
        g = G.grid
        eps = self.voxel_field(G, values, only_seen=False).reshape(g.n_r, g.n_phi, g.n_z)  # every voxel radiates, seen or not
        vol = g.r_centers[:, None, None] * np.diff(g.r_edges).mean() * np.diff(g.z_edges).mean() * (2.0 * np.pi / g.n_phi)
        return (eps * vol).sum(axis=(0, 2))

    def model_signals(self, G, values: dict) -> np.ndarray:
        """The channels' signals with every amplitude 1, on matrix G, in data order."""
        return self.design(G, values).sum(axis=1)

    def _scale_ref(self, d: np.ndarray, m: np.ndarray) -> float:
        """Absolute amplitude for a = 1: the model's peak scaled to the data's peak."""
        peak_m = float(np.max(np.abs(m))) if m.size else 0.0
        peak_d = float(np.max(np.abs(d))) if d.size else 0.0
        return peak_d / peak_m if peak_m > 0 and peak_d > 0 else 1.0

    def amp_bounds(self, name: str) -> tuple:
        """An amplitude's bounds: a toroidal floor (c, c_<component>) may vanish; the others take a_bounds."""
        if name == "c" or name.startswith("c_"):
            return (0.0, float(self.a_bounds[1]))
        return tuple(self.a_bounds)

    def _amplitudes(self, X: np.ndarray, d: np.ndarray, sig: np.ndarray) -> tuple[np.ndarray, float]:
        """The amplitudes (relative) solved exactly within their bounds, and the scale to absolute."""
        from scipy.optimize import lsq_linear

        ref = self._scale_ref(d, X.sum(axis=1))
        A = (X * ref) / sig[:, None]
        lo = np.array([self.amp_bounds(n)[0] for n in self.amp_names], dtype=float)
        hi = np.array([self.amp_bounds(n)[1] for n in self.amp_names], dtype=float)
        if X.shape[1] == 1:  # one amplitude: the bounded optimum is the clipped one
            denom = float(A[:, 0] @ A[:, 0])
            a = float(A[:, 0] @ (d / sig)) / denom if denom > 0 else 0.0
            return np.array([np.clip(a, lo[0], hi[0])]), ref
        res = lsq_linear(A, d / sig, bounds=(lo, hi))
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
    def search(self, seed: int = 0, maxiter: int = 60, popsize: int = 12, tol: float = 1e-3, progress=None,
               start: dict | None = None) -> ParametricResult:
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
        extra: dict = {"seed": seed}  # the stub does not list seed or x0
        if start:  # a known good point joins the initial population (e.g. a composite's components fitted alone)
            mid = (lo + hi) / 2.0
            x0 = np.array([float(start.get(n, mid[i])) for i, n in enumerate(self.names)], dtype=float)
            x0 = np.where(np.isfinite(x0), x0, mid)
            margin = 1e-9 * (hi - lo)  # strictly inside: SciPy rescales x0 and a value on a bound can round outside
            extra["x0"] = np.clip(x0, lo + margin, hi - margin)
        de = differential_evolution(objective, list(zip(lo, hi, strict=True)), maxiter=maxiter, popsize=popsize, tol=tol,
                                    polish=False, **cast(Any, extra))
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
            lo_a, hi_a = self.amp_bounds(name)
            pars.add(name, value=float(np.clip(a0, lo_a, hi_a)), min=lo_a, max=hi_a)

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
        # a periodic parameter's range ends (the tilt, a poloidal angle) are not limits in any meaningful sense
        flags = at_bounds(values, {**{k: v for k, v in self.bounds.items() if not is_periodic(k)},
                                   **{n: self.amp_bounds(n) for n in self.amp_names if not (n == "c" or n.startswith("c_"))}})
        # a floor at zero is an answer (none needed), not a limit; at its upper bound it is flagged
        flags += [f"{n} at its upper bound {self.amp_bounds(n)[1]:g}" for n in self.amp_names
                  if (n == "c" or n.startswith("c_")) and values.get(n, 0.0) >= self.amp_bounds(n)[1] * (1 - 1e-6)]
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



# ---------------------------------------------------------------------------
# Composites: two or more structures fitted together
# ---------------------------------------------------------------------------


class CompositeFit(ParametricFit):
    """
    Several models fitted together, each a component with its own shape
    parameters (prefixed by the component's name: band_psi_center) and its
    own amplitude(s) (a_band; per camera location with the step scale). All
    amplitudes are solved exactly at every trial; the search covers the
    shapes and the toroidal parameters.

    Each component has a toroidal peak location and a decay, in its own
    sense: a field-aligned component's source plane and decay along the
    field; an axisymmetric component's sector-scale centre and decay b.
    shared_toroidal: one phi_peak and one decay for every component;
    otherwise each has its own (band_decay, spot_phi_source, ring_b, ...).
    The peak location is fitted with vary_peak (within
    peak_shift_max_degrees of the injector), else fixed at the injector.

    components: [(name, model, template, bounds, fixed)]
    """

    def __init__(self, name: str, components: list, matrices: list, data: dict, scale_def: str, injection_deg: float,
                 shared_toroidal: bool = False, supersample=None, a_bounds: tuple | None = None, b_bounds: tuple | None = None,
                 vary_peak: bool = False, peak_shift_max_degrees: float = 180.0, toroidal_floor: bool = False):
        from emis3d.globals import AMPLITUDE_BOUNDS, DECAY_BOUNDS

        if scale_def not in SCALE_DEFS:
            raise ValueError(f"the parametric fit supports scale_def {', '.join(SCALE_DEFS)}, not {scale_def!r}")
        if len(components) < 2:
            raise ValueError("a composite needs at least two components")
        self.model = name
        self.matrices = list(matrices)
        self.scale_def = scale_def
        self.injection_deg = float(injection_deg)
        self.supersample = supersample
        self.shared = bool(shared_toroidal)
        self.template = components[0][2]  # for the machine (camera positions, wall)
        # amplitudes may vanish: a component the data do not need goes to 0 (and is flagged)
        hi = (a_bounds if a_bounds is not None else AMPLITUDE_BOUNDS)[1]
        self.a_bounds = (0.0, float(hi))
        decay_bounds = tuple(b_bounds if b_bounds is not None else DECAY_BOUNDS.get(scale_def, DECAY_BOUNDS["exponential"]))
        phi_bounds = (self.injection_deg - float(peak_shift_max_degrees), self.injection_deg + float(peak_shift_max_degrees))
        self.fixed: dict = {}
        self.bounds: dict = {}
        self.comps = []
        names_seen = set()
        for cname, model, template, cbounds, cfixed in components:
            if model not in MODELS:
                raise ValueError(f"component {cname}: unknown model {model!r}; choose from {sorted(MODELS)}")
            if not str(cname).isidentifier() or cname in names_seen:
                raise ValueError(f"component names must be distinct identifiers, got {cname!r}")
            names_seen.add(cname)
            spec = MODELS[model]
            defaults = default_bounds(template)
            comp = {"name": cname, "model": model, "spec": spec, "template": template, "emission": spec["emission"],
                    "field_aligned": bool(spec.get("field_aligned")), "shape": []}
            for pname, b in spec["params"].items():
                full = f"{cname}_{pname}"
                if pname in (cfixed or {}):
                    self.fixed[full] = float(cfixed[pname])
                else:
                    self.bounds[full] = tuple((cbounds or {}).get(pname) or b or defaults[pname])
                comp["shape"].append(pname)
            # --- toroidal parameters: shared, or the component's own
            has_decay = comp["field_aligned"] or scale_def in DECAY_BOUNDS
            if spec.get("fixed_plane"):  # in the injector's plane, whatever the others do
                comp["decay"] = ("decay" if self.shared else f"{cname}_decay") if has_decay else None
                comp["phi"] = None
                if not self.shared and comp["decay"]:
                    self.bounds[comp["decay"]] = tuple((cbounds or {}).get("decay") or decay_bounds)
            elif self.shared:
                comp["decay"] = "decay" if has_decay else None
                comp["phi"] = "phi_peak"
            else:
                comp["decay"] = (f"{cname}_decay" if comp["field_aligned"] else f"{cname}_b") if has_decay else None
                comp["phi"] = f"{cname}_phi_source" if comp["field_aligned"] else f"{cname}_phi_peak"
                if comp["decay"]:
                    self.bounds[comp["decay"]] = tuple((cbounds or {}).get("decay") or decay_bounds)
                if vary_peak:
                    self.bounds[comp["phi"]] = tuple((cbounds or {}).get("phi_source") or (cbounds or {}).get("phi_peak") or phi_bounds)
                else:
                    self.fixed[comp["phi"]] = self.injection_deg
            self.comps.append(comp)
        if self.shared:
            if any(c["decay"] for c in self.comps):
                self.bounds["decay"] = decay_bounds
            if any(c["phi"] for c in self.comps):
                if vary_peak:
                    self.bounds["phi_peak"] = phi_bounds
                else:
                    self.fixed["phi_peak"] = self.injection_deg
        self.names = list(self.bounds)

        # --- data aligned with each matrix's rows (as ParametricFit)
        self.data = data
        self._rows = {}
        for G in self.matrices:
            tags = {t: i for i, t in enumerate(G.channels)}
            keep = [t for t in data if np.isfinite(data[t][1]) and data[t][1] > 0]
            missing = sorted(set(keep) - set(tags))
            if missing:
                raise ValueError(f"channels not in the geometry matrix ({G.grid.describe()}): {missing}")
            self._rows[id(G)] = (np.array([tags[t] for t in keep]), np.array([data[t][0] for t in keep], dtype=float),
                                 np.array([data[t][1] for t in keep], dtype=float))
        self.n_data = len(self._rows[id(self.matrices[0])][0])

        # --- amplitude columns: one per component, or per camera location for an axisymmetric component under the step scale
        self.step_deg = []  # the composite's step lives in its components
        self._regions = {}
        step_locs: list = []
        if scale_def == "step":
            G0 = self.matrices[0]
            rows0 = self._rows[id(G0)][0]
            in_fit = {G0.cameras[i] for i in np.unique(G0.camera_of_row[rows0])}
            angles = {b.name: float(b.info["CAMERA_POSITION_R_Z_PHI"][2]) % 360.0 for b in self.template.tokamak.bolometers if b.name in in_fit}
            step_locs = sorted({round(a, 6) for a in angles.values()})
            locs = np.deg2rad(step_locs)
            for G in self.matrices:
                gap = np.abs((G.grid.phi_centers[:, None] - locs[None, :] + np.pi) % (2.0 * np.pi) - np.pi)
                self._regions[id(G)] = np.argmin(gap, axis=1)
        self.amp_names = []
        for comp in self.comps:
            if step_locs and not comp["field_aligned"]:
                comp["amps"] = [f"a_{comp['name']}_{int(round(d))}" for d in step_locs]
                comp["step_deg"] = step_locs
            else:
                comp["amps"] = [f"a_{comp['name']}"]
                comp["step_deg"] = []
            # an axisymmetric component with a peaked profile may have a uniform floor under it
            comp["floor"] = bool(toroidal_floor) and not comp["field_aligned"] and not comp["step_deg"] and bool(comp["decay"])
            if comp["floor"]:
                comp["amps"].append(f"c_{comp['name']}")
            self.amp_names += comp["amps"]
        self._f_cache = {}
        self.evaluations = 0
        self.has_decay = False
        self.field_aligned = False

    # ------------------------------------------------------------------
    def _value(self, values: dict, name: str | None, default: float = 0.0) -> float:
        if name is None:
            return default
        if name in values:
            return float(values[name])
        return float(self.fixed.get(name, default))

    def _configure(self, comp: dict, values: dict) -> tuple[float, float]:
        """Set a component's template to the trial's shape; return its (decay, peak angle [deg])."""
        t = comp["template"]
        for pname in comp["shape"]:
            full = f"{comp['name']}_{pname}"
            t.info[pname] = self._value(values, full, t.info.get(pname, 0.0))
        decay = self._value(values, comp["decay"], 0.0)
        phi = self._value(values, comp["phi"], self.injection_deg)
        if comp["field_aligned"]:
            t.info["decay"], t.info["phi_source"] = decay, phi
        return decay, phi

    def _apply(self, values: dict) -> None:
        for comp in self.comps:
            self._configure(comp, values)

    def voxel_columns(self, G, values: dict, only_seen: bool = True) -> list:
        cols = []
        sector = G.sector_of_voxel()
        for comp in self.comps:
            decay, phi = self._configure(comp, values)
            eps = G.emissivity_on_grid(comp["template"], comp["emission"], supersample=self.supersample, only_seen=only_seen)
            if comp["field_aligned"]:
                cols.append(eps)  # its decay and source plane are inside the mapped field
            elif comp["step_deg"]:
                region = self._regions[id(G)][sector]
                cols += [eps * (region == j) for j in range(len(comp["step_deg"]))]
            elif comp["decay"]:
                f = np.asarray(Util_emis3D.scale_wrapper(a=1.0, b=decay, phi=G.grid.phi_centers, mu=float(np.deg2rad(phi)),
                                                         scale_def=self.scale_def, emissionName=comp["emission"]), dtype=float)
                cols.append(eps * f[sector])
                if comp["floor"]:
                    cols.append(eps)  # the uniform floor
            else:
                cols.append(eps)
        return cols

    def design(self, G, values: dict) -> np.ndarray:
        rows, _, _ = self._rows[id(G)]
        self.evaluations += 1
        return np.column_stack([G.signals(c)[rows] for c in self.voxel_columns(G, values)])

    def start_from(self, single_results: dict) -> dict:
        """
        A starting point from the components' models fitted alone (model -> ParametricResult):
        each component's shape and toroidal parameters, where its model was fitted.
        """
        start: dict = {}
        for comp in self.comps:
            res = single_results.get(comp["model"])
            if res is None:
                continue
            for pname in comp["shape"]:
                if pname in res.values:
                    start[f"{comp['name']}_{pname}"] = res.values[pname]
            own_decay = res.values.get("decay", res.values.get("b"))
            if comp["decay"] and own_decay is not None and comp["decay"] not in start:
                start[comp["decay"]] = own_decay
            own_phi = res.values.get("phi_source")
            if own_phi is not None and comp["phi"] and comp["phi"] not in start:
                start[comp["phi"]] = own_phi
        return start

    def search(self, *args, **kwargs) -> ParametricResult:
        result = super().search(*args, **kwargs)
        comps = []
        for comp in self.comps:
            vals: dict[str, Any] = {p: self._value(result.values, f"{comp['name']}_{p}") for p in comp["shape"]}
            vals["decay"] = self._value(result.values, comp["decay"]) if comp["decay"] else None
            vals["phi"] = self._value(result.values, comp["phi"], self.injection_deg)
            comps.append({"name": comp["name"], "model": comp["model"], "values": vals,
                          "amplitudes": {n: result.values[n] for n in comp["amps"]}, "field_aligned": comp["field_aligned"]})
        result.components = comps
        return result
