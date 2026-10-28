# uncertainty.py
"""
Uncertainty ranges for quantities derived from an Emis3D fit: toroidal peaking
factor (TPF), total radiated power, peak location, peak power, power fraction
near the peak, amplitude and toroidal decay lengths.

Three sources are combined, and each can be reported on its own:

1. statistical  - parameter covariance of the best-fit radDist
2. model        - other radDists that fit almost as well, weighted by Akaike
                  weights from the same score that ranks the fits
                  (Util_emis3D.fit_score, lmfit's AIC)
3. calibration  - relative calibration uncertainty per bolometer group; each
                  draw rescales the data and refits the candidate radDists

The toroidal profile is computed by ``toroidal_profile``, which
Emis3D._post_process_calculations also uses, so the best-fit numbers and the
ranges always come from the same code.

The per-camera calibration uncertainties can be taken from
`emis3d calibrate`, which prints a ready-made block.

Run-config block (all keys optional):

    uncertainty:
      enabled: true
      n_samples: 2000          # parameter samples, pooled over all sources
      max_candidates: 25       # radDists considered for the model term
      min_weight: 1.0e-3       # drop candidates below this relative weight
      calibration:             # relative 1-sigma calibration error per group
        SX90PF: 0.10
        DISRADU: 0.10
      n_calibration: 20        # calibration draws (0 disables the term)
      window_deg: 45.0         # half-width for the power-fraction quantity
      seed: 0
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import emis3d.util_emis3D as Util_emis3D

logger = logging.getLogger(__name__)

TWO_PI = 2.0 * np.pi

# Quantities reported for every sample; units in QUANTITY_UNITS
QUANTITIES = (
    "tpf",
    "total_power",
    "peak_phi_deg",
    "peak_power",
    "power_fraction_window",
    "amplitude",
)
QUANTITY_UNITS = {
    "tpf": "",
    "total_power": "P_pol units x scaleSynth",
    "peak_phi_deg": "deg",
    "peak_power": "P_pol units x scaleSynth per rad",
    "power_fraction_window": "",
    "amplitude": "",
    "decay_length_deg": "deg",
}


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------


@dataclass
class UncertaintyConfig:
    enabled: bool = True
    n_samples: int = 2000
    max_candidates: int = 25
    min_weight: float = 1.0e-3
    calibration: dict = field(default_factory=dict)
    n_calibration: int = 0
    window_deg: float = 45.0
    seed: int | None = 0
    intervals: tuple = (0.68, 0.95)

    @classmethod
    def from_info(cls, info: dict | None) -> "UncertaintyConfig":
        block = dict((info or {}).get("uncertainty", {}) or {})
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(block) - known
        if unknown:
            raise ValueError(f"Unknown uncertainty options: {sorted(unknown)}")
        cfg = cls(**block)
        if cfg.calibration and cfg.n_calibration == 0:
            cfg.n_calibration = 20
        return cfg


# ----------------------------------------------------------------------------
# Toroidal profile and derived quantities
# ----------------------------------------------------------------------------


def _peak_location(synth: dict, params: dict) -> float:
    mu = float(synth["injectionLocation_rad"])
    if "peak_rad_loc" in params:
        mu = float(params["peak_rad_loc"])
    return mu


def toroidal_profile(
    params: dict,
    synth: dict,
    scale_def: str,
    bolo_phi_locs: np.ndarray | None = None,
) -> dict:
    """
    Radiated power per radian around the torus for one parameter set.

    P(phi) = sum_e  P_pol_e(phi) * scale_e(phi) * scaleSynth_e

    Uses the raw peak location (not snapped to the grid) so it matches the
    residual used in the fit. Requires 'phi_array' and 'P_pol' for every
    emission (threaded into synthetic_dict by Emis3D._build_synthetic_dict).
    """
    tag = Util_emis3D.loc_tag(synth["injectionLocation"])
    mu = _peak_location(synth, params)
    a = float(params[f"a_{tag}"])

    phi = None
    total = None
    per_emission = {}
    for name in synth["emissionNames"]:
        sub = synth[name]
        phi_e = np.asarray(sub["phi_array"], dtype=float)
        P_pol = np.asarray(sub["P_pol"], dtype=float)
        if phi is None:
            phi = phi_e
        elif not np.allclose(phi_e, phi):
            raise ValueError("phi_array differs between emission components")

        b = float(params.get(f"b_{name}_{tag}", 0.0))
        dphi = Util_emis3D.find_dphi(phi, mu, emissionName=name)
        scale = Util_emis3D.scale_wrapper(
            a=a,
            b=b,
            phi=phi,
            dphi=dphi,
            mu=mu,
            scale_def=scale_def,
            emissionName=name,
            bolo_phi_locs=bolo_phi_locs,
            c=float(params.get(f"c_{tag}", 0.0)),  # the toroidal floor, when fitted
        )
        P_e = P_pol * np.asarray(scale, dtype=float) * float(sub["scaleSynth"])
        per_emission[name] = {"multiplication_factor": np.asarray(scale), "power": P_e}
        total = P_e if total is None else total + P_e

    return {"phi": phi, "total_power": total, "per_emission": per_emission, "mu": mu}


def _decay_length_deg(scale_def: str, a: float, b: float) -> float:
    """Toroidal distance (deg) over which the scale factor falls by 1/e (or to 0)."""
    if scale_def == "exponential":
        return float(np.rad2deg(1.0 / np.sqrt(b))) if b > 0 else np.inf
    if scale_def == "linear":
        return float(np.rad2deg(a / abs(b))) if b != 0 else np.inf
    return np.inf


def derived_quantities(
    params: dict,
    synth: dict,
    scale_def: str,
    bolo_phi_locs: np.ndarray | None = None,
    window_deg: float = 45.0,
) -> tuple[dict, np.ndarray]:
    """Scalar summaries of the toroidal profile, plus the profile itself."""
    prof = toroidal_profile(params, synth, scale_def, bolo_phi_locs)
    phi, P = prof["phi"], prof["total_power"]
    dphi_grid = TWO_PI / phi.size  # uniform periodic grid from linspace(endpoint=False)

    mean = float(np.mean(P))
    k = int(np.argmax(P))
    # sub-grid peak location: parabola through the maximum and its periodic neighbours
    y0, y1, y2 = P[k - 1], P[k], P[(k + 1) % P.size]
    denom = y0 - 2.0 * y1 + y2
    shift = 0.5 * (y0 - y2) / denom if denom < 0 else 0.0
    peak_phi = phi[k] + float(np.clip(shift, -0.5, 0.5)) * dphi_grid
    dist = np.abs((phi - phi[k] + np.pi) % TWO_PI - np.pi)
    in_window = dist <= np.deg2rad(window_deg)

    tag = Util_emis3D.loc_tag(synth["injectionLocation"])
    out = {
        "tpf": float(P[k] / mean) if mean > 0 else np.nan,
        "total_power": float(P.sum() * dphi_grid),
        "peak_phi_deg": float(np.rad2deg(peak_phi) % 360.0),
        "peak_power": float(P[k]),
        "power_fraction_window": float(P[in_window].sum() / P.sum()) if P.sum() > 0 else np.nan,
        "amplitude": float(params[f"a_{tag}"]),
    }
    if f"c_{tag}" in params:  # the toroidal floor, as a fraction of the peak amplitude
        out["floor_fraction"] = float(params[f"c_{tag}"]) / out["amplitude"] if out["amplitude"] > 0 else np.nan
    for name in synth["emissionNames"]:
        b = float(params.get(f"b_{name}_{tag}", 0.0))
        out[f"decay_length_deg[{name}]"] = _decay_length_deg(scale_def, out["amplitude"], b)
    return out, P


# ----------------------------------------------------------------------------
# Parameter sampling
# ----------------------------------------------------------------------------


def ensure_covariance(fit, data_dict: dict, synth: dict, scale_def: str, helical_endpoint_weight: float = 0.0):
    """
    Return a fit with a usable covariance matrix. If the fit has none (for
    example differential evolution without numdifftools), refit locally with
    least_squares from the best-fit values. The refit is only accepted if it
    is at least as good.
    """
    covar = getattr(fit, "covar", None)
    if covar is not None and np.all(np.isfinite(covar)):
        return fit, "ok"
    try:
        refit = Util_emis3D.fit_radDist(fit.params, data_dict, synth, scale_def, helical_endpoint_weight, method="least_squares")
    except Util_emis3D.FIT_ERRORS as e:
        logger.warning("Covariance refit failed: %s", e)
        return fit, "no covariance"
    refit_covar = getattr(refit, "covar", None)
    if refit_covar is not None and np.all(np.isfinite(refit_covar)) and refit.chisqr <= fit.chisqr * (1 + 1e-6):
        return refit, "refit"
    return fit, "no covariance"


def sample_parameters(fit, n: int, rng: np.random.Generator):
    """
    Draw parameter sets from the fit's (redchi-scaled) covariance, held within
    the parameter bounds (draws beyond a bound are placed on it). Returns
    (list of value dicts, flags).
    """
    base = fit.params.valuesdict()
    flags = set()
    names = list(fit.var_names or [])
    covar = getattr(fit, "covar", None)

    for nm in names:
        p = fit.params[nm]
        span = (p.max - p.min) if np.isfinite(p.max - p.min) else np.inf
        tol = 1e-3 * span if np.isfinite(span) else 1e-9
        if np.isfinite(p.min) and p.value - p.min <= tol:
            flags.add(f"{nm} at lower bound")
        if np.isfinite(p.max) and p.max - p.value <= tol:
            flags.add(f"{nm} at upper bound")

    if n <= 0:
        return [], flags
    if not names or covar is None or not np.all(np.isfinite(covar)):
        flags.add("no covariance: statistical spread not included")
        return [dict(base) for _ in range(n)], flags

    mean = np.array([fit.params[nm].value for nm in names])
    lo = np.array([fit.params[nm].min for nm in names])
    hi = np.array([fit.params[nm].max for nm in names])
    cov = 0.5 * (covar + covar.T)

    # Draws beyond a bound are moved onto it (clipped), not rejected: a bound is a
    # hard limit, so the probability beyond it belongs on it. Rejecting them made
    # the range of a parameter sitting on a bound one-sided, so the range could
    # miss the best fit itself (a TPF of 1.00 with a "68% range" of 1.02-1.13).
    x = rng.multivariate_normal(mean, cov, size=n, method="svd")
    outside = np.any((x < lo) | (x > hi), axis=1)
    if outside.mean() > 0.25:
        flags.add(f"{100 * outside.mean():.0f}% of draws reached a bound and were held on it")
    accepted = list(np.clip(x, lo, hi))

    # Set the varied parameters on a copy so lmfit recomputes the derived ones
    # (peak_rad_loc = (injector_phi + peak_shift) % 2pi), which would otherwise
    # stay at their best-fit values and collapse the peak location's range
    work = fit.params.copy()
    samples = []
    for row in accepted[:n]:
        for nm, v in zip(names, row, strict=True):
            work[nm].value = float(v)
        samples.append(dict(work.valuesdict()))
    return samples, flags


# ----------------------------------------------------------------------------
# Candidate weighting and calibration perturbation
# ----------------------------------------------------------------------------


def candidate_weights(fits: list) -> tuple[np.ndarray, float]:
    """
    Akaike weights from the score that ranks the fits (Util_emis3D.fit_score,
    lmfit's AIC = N ln(chi2/N) + 2k, independent of the error model's scale):

        w_i ~ exp(-(AIC_i - AIC_min) / 2)

    Returns (weights, best score).
    """
    score = np.array([Util_emis3D.fit_score(f) for f in fits], dtype=float)
    if not np.any(np.isfinite(score)):
        raise RuntimeError("No candidate fit has a finite score")
    best = float(np.min(score[np.isfinite(score)]))
    w = np.where(np.isfinite(score), np.exp(-0.5 * (score - best)), 0.0)
    return w / w.sum(), best


def perturb_calibration(data_dict: dict, bolometer_order: list, calibration: dict, rng) -> tuple[dict, dict]:
    """Scale each bolometer group's data and error by an independent log-normal factor."""
    new = copy.deepcopy(data_dict)
    factors = {}
    for ii, bolo in enumerate(bolometer_order):
        sigma = float(calibration.get(bolo, 0.0))
        f = float(np.exp(rng.normal(0.0, sigma))) if sigma > 0 else 1.0
        factors[bolo] = f
        new["observed"][ii] = [v * f for v in new["observed"][ii]]
        new["observed_error"][ii] = [e * f for e in new["observed_error"][ii]]
    return new, factors


# ----------------------------------------------------------------------------
# Propagation
# ----------------------------------------------------------------------------


@dataclass
class UncertaintyResult:
    samples: dict              # quantity -> array over pooled samples
    source: np.ndarray         # 0 best fit only, 1 nominal mixture, 2 calibration draws
    candidate: np.ndarray      # candidate index of each sample
    phi: np.ndarray
    profiles: np.ndarray       # (n_samples, n_phi) toroidal profiles
    weights: np.ndarray        # nominal candidate weights
    candidate_ids: list
    best_score: float          # AIC of the best candidate
    flags: dict                # candidate id -> sorted list of flags
    calibration_factors: list = field(default_factory=list)
    config: UncertaintyConfig | None = None
    best_values: dict = field(default_factory=dict)  # the best candidate's own derived quantities (no sampling)
    best_profile: np.ndarray | None = None  # its toroidal profile

    def summary(self, which: str = "total") -> dict:
        """
        Median and central intervals per quantity for 'total', 'statistical'
        or 'model'. Every interval contains the best fit's own value: draws
        held on parameter bounds, or a wide (reduced-chi2-scaled)
        covariance, can put most of the draws on one side of it.
        """
        mask = {
            "statistical": self.source == 0,          # best radDist, parameter covariance
            "model": self.source == 1,                # + other radDists that fit nearly as well
            "total": self.source >= 1,                # + calibration draws, if configured
        }[which]
        out = {}
        for q, vals in self.samples.items():
            v = np.asarray(vals)[mask]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            entry = {"median": float(np.median(v)), "n": int(v.size)}
            anchor = self.best_values.get(q)
            anchor = float(anchor) if anchor is not None and np.isfinite(anchor) else None
            for c in (self.config or UncertaintyConfig()).intervals:
                lo, hi = np.percentile(v, [50 * (1 - c), 50 * (1 + c)])
                if anchor is not None:
                    lo, hi = min(lo, anchor), max(hi, anchor)
                entry[f"{int(round(100 * c))}%"] = (float(lo), float(hi))
            out[q] = entry
        return out

    def profile_band(self, levels=(0.68, 0.95)) -> dict:
        prof = self.profiles[self.source >= 1]
        band: dict = {"phi": self.phi, "median": np.median(prof, axis=0)}
        for c in levels:
            lo, hi = np.percentile(prof, [50 * (1 - c), 50 * (1 + c)], axis=0)
            if self.best_profile is not None and self.best_profile.shape == lo.shape:  # the band contains the best fit
                lo, hi = np.minimum(lo, self.best_profile), np.maximum(hi, self.best_profile)
            band[f"{int(round(100 * c))}%"] = (lo, hi)
        return band

    def as_dict(self) -> dict:
        """Compact, pickle-friendly record for bestFits."""
        return {
            "total": self.summary("total"),
            "statistical": self.summary("statistical"),
            "model": self.summary("model"),
            "profile_band": self.profile_band(),
            "candidates": [
                {"fit_id": cid, "weight": float(w), "flags": self.flags.get(cid, [])}
                for cid, w in zip(self.candidate_ids, self.weights, strict=True)
            ],
            "best_score": self.best_score,
            "calibration_factors": self.calibration_factors,
            "units": QUANTITY_UNITS,
        }


def propagate(
    candidates: list,
    data_dict: dict,
    scale_def: str,
    bolometer_order: list,
    cfg: UncertaintyConfig,
    bolo_phi_locs: np.ndarray | None = None,
    helical_endpoint_weight: float = 0.0,
) -> UncertaintyResult:
    """
    candidates : list of (fit_id, lmfit MinimizerResult, synthetic_dict), best first
    data_dict  : Emis3D.fitData[evalTime] (needs 'observed' and 'observed_error')
    """
    rng = np.random.default_rng(cfg.seed)
    n_real = 1 + (cfg.n_calibration if cfg.calibration else 0)
    per_real = max(1, cfg.n_samples // n_real)

    # --- nominal fits with covariance
    nominal = []
    flags = {}
    for cid, fit, synth in candidates:
        fit2, status = ensure_covariance(fit, data_dict, synth, scale_def, helical_endpoint_weight)
        if status == "no covariance":
            flags.setdefault(cid, set()).add("no covariance: statistical spread not included")
        nominal.append((cid, fit2, synth))

    weights, best_score = candidate_weights([f for _, f, _ in nominal])
    keep = weights >= cfg.min_weight * weights.max()
    nominal = [c for c, k in zip(nominal, keep, strict=True) if k]
    weights = weights[keep] / weights[keep].sum()

    phi_ref = None
    rows, profiles, source, cand_idx = [], [], [], []
    cal_factors = []

    def add_samples(fit_list, counts, src):
        nonlocal phi_ref
        for j, ((cid, fit, synth), n_j) in enumerate(zip(fit_list, counts, strict=True)):
            if n_j == 0:
                continue
            samples, fl = sample_parameters(fit, int(n_j), rng)
            if fl:
                flags.setdefault(cid, set()).update(fl)
            for pv in samples:
                q, P = derived_quantities(pv, synth, scale_def, bolo_phi_locs, cfg.window_deg)
                rows.append(q)
                profiles.append(P)
                source.append(src)
                cand_idx.append(j)
            if phi_ref is None:
                phi_ref = np.asarray(synth[synth["emissionNames"][0]]["phi_array"], dtype=float)

    # --- source 0: best candidate only (statistical term)
    best = int(np.argmax(weights))
    cid_b, fit_b, synth_b = nominal[best]
    best_values, best_profile = derived_quantities(fit_b.params.valuesdict(), synth_b, scale_def, bolo_phi_locs, cfg.window_deg)
    counts = np.zeros(len(nominal), int)
    counts[best] = per_real
    add_samples(nominal, counts, 0)

    # --- source 1: nominal data, all candidates by weight (statistical + model)
    add_samples(nominal, rng.multinomial(per_real, weights), 1)

    # --- source 2: calibration draws, refit and reweighted each time
    for _ in range(n_real - 1):
        dd, factors = perturb_calibration(data_dict, bolometer_order, cfg.calibration, rng)
        cal_factors.append(factors)
        fits_r = []
        for cid, fit, synth in nominal:
            try:
                refit = Util_emis3D.fit_radDist(fit.params, dd, synth, scale_def, helical_endpoint_weight, method="least_squares")
            except Util_emis3D.FIT_ERRORS as e:
                logger.warning("Calibration refit failed for fit %s: %s", cid, e)
                refit = fit
            fits_r.append((cid, refit, synth))
        w_r, _ = candidate_weights([f for _, f, _ in fits_r])
        add_samples(fits_r, rng.multinomial(per_real, w_r), 2)

    if phi_ref is None:
        raise RuntimeError("No parameter samples were drawn; check n_samples and the candidate weights")
    keys = sorted({k for row in rows for k in row})
    samples = {k: np.array([row.get(k, np.nan) for row in rows]) for k in keys}

    return UncertaintyResult(
        samples=samples,
        source=np.array(source),
        candidate=np.array(cand_idx),
        phi=phi_ref,
        profiles=np.array(profiles),
        weights=weights,
        candidate_ids=[cid for cid, _, _ in nominal],
        best_score=best_score,
        flags={cid: sorted(v) for cid, v in flags.items()},
        calibration_factors=cal_factors,
        config=cfg,
        best_values={k: float(v) for k, v in best_values.items() if isinstance(v, (int, float, np.floating))},
        best_profile=np.asarray(best_profile, dtype=float) if best_profile is not None else None,
    )


def format_summary(result: UncertaintyResult, which: str = "total") -> str:
    """Human-readable table for logs and reports."""
    lines = [f"{'quantity':34s} {'median':>12s} {'68% interval':>27s} {'95% interval':>27s}"]
    for q, e in result.summary(which).items():
        lo68, hi68 = e["68%"]
        lo95, hi95 = e["95%"]
        lines.append(f"{q:34s} {e['median']:12.4g} [{lo68:11.4g}, {hi68:11.4g}] [{lo95:11.4g}, {hi95:11.4g}]")
    return "\n".join(lines)
