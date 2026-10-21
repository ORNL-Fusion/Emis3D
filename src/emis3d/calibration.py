# calibration.py
"""
Cross-calibration of bolometer/SXR cameras.

During the current quench of a disruption the radiation is assumed to be
axisymmetric, so every camera should see the same radDist with the same
amplitude. Fitting each camera's amplitude separately, the ratios between
those amplitudes are the calibration corrections.

With axisymmetric radDists the model for camera g is c_g * s_g, which is
linear in c_g, so for each candidate radDist the best amplitude has a closed
form (weighted least squares over the camera's channels):

    c_g = sum(w^2 d s) / sum(w^2 s^2),     w = 1 / sigma

No optimizer, bounds or starting values are needed. The radDist with the
lowest total chi-square is chosen at each evaluation time, and each camera's
correction factor is c_ref / c_g: multiplying camera g's data by it brings it
in line with the reference camera.

Use it from the command line,

    emis3d calibrate runs/DIII-D/184407/crossCalib.yaml --times 2120:2128:0.5

or from Python with cross_calibrate(). The run config is a normal one: its
radDistDirectories_LocIndependent should hold axisymmetric radDists (for
example elongated rings); its scale_def is ignored, because axisymmetry is
the method.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    cameras: list[str]
    times: list[float]
    reference: str
    amplitudes: np.ndarray          # (n_times, n_cameras)
    amplitude_errors: np.ndarray    # (n_times, n_cameras)
    factors: np.ndarray             # (n_times, n_cameras): c_ref / c_g
    factor_errors: np.ndarray       # (n_times, n_cameras), statistical
    best_raddist: list[str]         # per time
    chi2: list[float]               # per time, best radDist
    current_scaling: dict = field(default_factory=dict)
    data_files: dict = field(default_factory=dict)

    @property
    def factor(self) -> dict:
        """Median correction factor per camera over the evaluation times."""
        return {c: float(np.nanmedian(self.factors[:, j])) for j, c in enumerate(self.cameras)}

    @property
    def spread(self) -> dict:
        """Relative spread (standard deviation / median) of each factor over time."""
        out = {}
        for j, c in enumerate(self.cameras):
            f = self.factors[:, j]
            f = f[np.isfinite(f)]
            out[c] = float(np.std(f, ddof=1) / np.median(f)) if f.size > 1 else float("nan")
        return out

    @property
    def statistical_error(self) -> dict:
        """Median relative statistical error of each factor."""
        return {c: float(np.nanmedian(self.factor_errors[:, j] / self.factors[:, j]))
                for j, c in enumerate(self.cameras)}

    @property
    def new_scaling(self) -> dict:
        """scalingFactor to put in run configs: current value times the correction."""
        return {c: self.current_scaling.get(c, 1.0) * self.factor[c] for c in self.cameras}

    def table(self) -> str:
        lines = [
            f"Cross-calibration over {len(self.times)} time(s), reference camera {self.reference}",
            f"{'camera':12s} {'factor':>8s} {'spread':>8s} {'stat err':>9s} {'scalingFactor':>14s}",
        ]
        for c in self.cameras:
            spread = self.spread[c]
            lines.append(
                f"{c:12s} {self.factor[c]:8.4f} "
                f"{(f'{100 * spread:7.2f}%' if np.isfinite(spread) else '      -'):>8s} "
                f"{100 * self.statistical_error[c]:8.2f}% "
                f"{self.current_scaling.get(c, 1.0):6.3f} -> {self.new_scaling[c]:.4g}"
            )
        best = sorted(set(self.best_raddist))
        lines.append(f"best radDist(s): {', '.join(best)}")
        return "\n".join(lines)

    def yaml_block(self) -> str:
        """A BOLOMETERS block with the new scaling factors, ready to paste into run configs."""
        out = [
            f"# Cross-calibrated against {self.reference} over {len(self.times)} time(s), "
            f"{min(self.times):g}-{max(self.times):g}",
            "BOLOMETERS:",
        ]
        for c in self.cameras:
            spread = self.spread[c]
            note = f"spread {100 * spread:.1f}%" if np.isfinite(spread) else "single time"
            out.append(f"  {c}:")
            if c in self.data_files:
                out.append(f"    dataFileName: '{self.data_files[c]}'")
            out.append(f"    scalingFactor: {self.new_scaling[c]:.4g}  # x{self.factor[c]:.4f}, {note}")
        return "\n".join(out) + "\n"


def camera_amplitude(data, error, synth) -> tuple[float, float, float, int]:
    """
    Weighted least-squares amplitude c minimizing sum(((d - c s) / sigma)^2).

    Channels with non-positive or non-finite data are left out, as the fit's
    residual does, and so are dead channels (marked with DEAD_CHANNEL_ERROR by
    Emis3D's data preparation). Returns (c, sigma_c, chi2, n_channels_used); sigma_c is
    scaled by the camera's reduced chi-square, so it reflects the actual
    scatter rather than the error model's arbitrary normalization.
    """
    d = np.asarray(data, dtype=float)
    e = np.asarray(error, dtype=float)
    s = np.asarray(synth, dtype=float)
    from emis3d.emis3D import DEAD_CHANNEL_ERROR

    use = np.isfinite(d) & np.isfinite(e) & np.isfinite(s) & (d > 0) & (e > 0) & (e < DEAD_CHANNEL_ERROR)
    if not use.any() or not np.any(s[use] != 0):
        return float("nan"), float("nan"), 0.0, 0
    w2 = 1.0 / e[use] ** 2
    sw = np.sum(w2 * s[use] ** 2)
    c = float(np.sum(w2 * d[use] * s[use]) / sw)
    chi2 = float(np.sum(w2 * (d[use] - c * s[use]) ** 2))
    n = int(use.sum())
    redchi = chi2 / (n - 1) if n > 1 else 1.0
    return c, float(np.sqrt(redchi / sw)), chi2, n


def _channel_totals(synth_group: list) -> np.ndarray:
    """One synthetic value per channel: the sum over its toroidal segments."""
    return np.array([float(np.sum(np.atleast_1d(ch))) for ch in synth_group])


def _raddist_label(rd) -> str:
    path = rd.info.get("radDistPath") if rd.info else None
    return Path(path).name if path else str(rd.info.get("distType", "radDist"))


def cross_calibrate(
    runConfig: str | os.PathLike,
    times,
    reference: str | None = None,
    workspace=None,
) -> CalibrationResult:
    """
    Cross-calibrate the cameras in a run config at one or more evaluation times.

    runConfig : run config path (or name inside the workspace's runs location)
    times     : evaluation times, in the units of the data
    reference : camera every factor is relative to; by default the camera with
                the largest median amplitude
    """
    from emis3d.emis3D import Emis3D

    times = [float(t) for t in np.atleast_1d(times)]
    if not times:
        raise ValueError("Give at least one evaluation time")

    e = Emis3D(runConfigName=str(runConfig), workspace=workspace)
    if e.info["radDistDirectories_LocDependent"]:
        raise ValueError(
            "Cross-calibration assumes axisymmetric radiation: list the axisymmetric "
            "radDists under radDistDirectories_LocIndependent and leave "
            "radDistDirectories_LocDependent empty"
        )
    if e.info.get("scale_def") != "constant":
        logger.info("Cross-calibration ignores scale_def (%s): it assumes axisymmetry", e.info.get("scale_def"))

    cameras = list(e.channel_order["bolometer_order"])
    nT, nC = len(times), len(cameras)
    amp = np.full((nT, nC), np.nan)
    amp_err = np.full((nT, nC), np.nan)
    best_label, best_chi2 = [], []

    for it, t in enumerate(times):
        e._prepare_fits(evalTime=t)
        observed = e.fitData[t]["observed"]
        observed_error = e.fitData[t]["observed_error"]

        best = None
        for k, entry in e.fits[t].items():
            if not isinstance(k, (int, np.integer)):
                continue
            sd = entry["synthetic_dict"]
            chi2_total, cams = 0.0, []
            for g in range(nC):
                synth = sum(_channel_totals(sd[name]["data"][g]) for name in sd["emissionNames"])
                c, sc, chi2, n = camera_amplitude(observed[g], observed_error[g], synth)
                cams.append((c, sc))
                chi2_total += chi2
            if best is None or chi2_total < best[0]:
                info = entry["info"]["radDists"]
                rd = e.data["synthetic"][info["locationdependence"]][info["radDistNumber"]]
                best = (chi2_total, cams, _raddist_label(rd))

        if best is None:
            raise RuntimeError(f"No radDists to calibrate against at t={t}")
        best_chi2.append(best[0])
        best_label.append(best[2])
        for g, (c, sc) in enumerate(best[1]):
            amp[it, g], amp_err[it, g] = c, sc

    # --- Reference camera and factors c_ref / c_g
    if reference is not None and reference not in cameras:
        raise ValueError(f"reference camera {reference!r} is not one of {cameras}")
    ref: str = reference if reference is not None else cameras[int(np.nanargmax(np.nanmedian(amp, axis=0)))]
    r = cameras.index(ref)
    factors = amp[:, [r]] / amp
    rel = np.sqrt((amp_err[:, [r]] / amp[:, [r]]) ** 2 + (amp_err / amp) ** 2)
    rel[:, r] = 0.0
    for g, c in enumerate(cameras):
        if not np.all(np.isfinite(amp[:, g])):
            logger.warning("Camera %s had no usable channels at some times", c)

    bolos = e.info["BOLOMETERS"]
    return CalibrationResult(
        cameras=cameras,
        times=times,
        reference=ref,
        amplitudes=amp,
        amplitude_errors=amp_err,
        factors=factors,
        factor_errors=factors * rel,
        best_raddist=best_label,
        chi2=best_chi2,
        current_scaling={c: float(bolos[c].get("scalingFactor", 1.0)) for c in cameras if c in bolos},
        data_files={c: bolos[c]["dataFileName"] for c in cameras if c in bolos and "dataFileName" in bolos[c]},
    )


def parse_times(tokens) -> list[float]:
    """'2124', '2120:2128:0.5' (inclusive) or a mix of both."""
    out = []
    for tok in tokens:
        if ":" in str(tok):
            parts = [float(x) for x in str(tok).split(":")]
            if len(parts) != 3 or parts[2] <= 0:
                raise ValueError(f"time range {tok!r} must be start:stop:step with step > 0")
            start, stop, step = parts
            n = int(np.floor((stop - start) / step + 1e-9)) + 1
            out.extend(float(start + i * step) for i in range(max(n, 0)))
        else:
            out.append(float(tok))
    return out
