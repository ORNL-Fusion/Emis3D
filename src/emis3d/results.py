# results.py
"""
The results of a run as plain numbers: one JSON file beside the dill file.

The dill file (Emis3D._save_bestFits) holds the live objects of a run: the
lmfit result, the rebuilt radDist with its tokamak and ray-tracing scene. It
is what Emis3D._load_bestFits and the figures need, but it can only be read
by a Python with the same Emis3D, lmfit, Cherab and Raysect, and most of
it is the scene (about 0.6 MB per time, of which the numbers are 20 kB).

This file holds what an analysis usually wants afterwards, in a form that
any program reads and that stays readable when the code changes: for every
time, the chosen radDist, the fitted parameters with their errors and
bounds, chi-square and AIC, each camera's channels with data, error and
model, the toroidal profile, the TPF and the ranges.

    from emis3d.results import load_summary
    r = load_summary("runs/184407/results/<run_id>/184407_bestFits_2100.0000.json")
    r["results"][0]["tpf"], r["results"][0]["fit"]["parameters"]["a_225"]["value"]

It is strict JSON: values that are not finite (the error of a channel left
out of the fit is infinite) are written as null.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from pathlib import Path

import numpy as np

FORMAT = 1


def to_plain(obj):
    """
    obj as JSON can hold it: arrays and tuples as lists, numpy scalars as
    Python numbers, non-finite numbers as None, paths as strings, dictionary
    keys as strings. Anything else (an object with no plain form) is None.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return to_plain(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, os.PathLike):
        return str(obj)
    return None


def _simple(info: dict) -> dict:
    """The entries of a settings dictionary that are single values (numbers, strings, booleans)."""
    return {k: v for k, v in (info or {}).items() if isinstance(v, (bool, int, float, str, np.integer, np.floating)) or v is None}


def _parameters(fit) -> dict:
    out = {}
    for name, p in fit.params.items():
        out[name] = {"value": p.value, "stderr": p.stderr, "min": p.min, "max": p.max, "vary": bool(p.vary), "expr": p.expr}
    return out


def _parametric(best: dict) -> dict | None:
    result = best.get("parametric")
    if result is None:
        return None
    out = {k: getattr(result, k, None) for k in ("model", "values", "stderr", "chi2", "n_data", "aic", "n_params", "at_bounds", "amplitude", "toroidal",
                                                  "components")}
    out["compared"] = [{"model": r.model, "aic": r.aic, "chi2": r.chi2, "n_params": r.n_params} for r in best.get("parametric_models", [])]
    return out


def time_summary(emis, evalTime: float) -> dict:
    """One time's results as plain values (see the module's description). Parts a run does not have are left out."""
    best = emis.bestFits[evalTime]
    fit_data = getattr(emis, "fitData", {}).get(evalTime, {})
    out: dict = {"time": evalTime}

    info = best.get("radDistInfo") or {}
    out["raddist"] = {"file": info.get("radDistPath"), "distType": info.get("distType"), "settings": _simple(info)}

    fit = best.get("fit")
    if fit is not None:
        out["fit"] = {"chi2": fit.chisqr, "reduced_chi2": fit.redchi, "aic": fit.aic, "bic": fit.bic, "n_data": fit.ndata, "n_varied": fit.nvarys,
                      "method": getattr(fit, "method", None), "success": bool(getattr(fit, "success", True)), "parameters": _parameters(fit)}
    out["at_bounds"] = list(best.get("at_bounds", []))

    # --- Each camera's channels: data, error (null: not in the fit) and the fitted model
    order = getattr(emis, "channel_order", None) or {"bolometer_order": [], "channel_list": []}
    synth = best.get("synthData") or {}
    cameras = {}
    for name, channels in zip(order["bolometer_order"], order["channel_list"], strict=True):
        if name not in fit_data.get("boloData", {}):
            continue
        data = np.asarray(fit_data["boloData"][name], dtype=float)
        error = np.asarray(fit_data["boloData_error"][name], dtype=float)
        cam = {"channels": [str(c) for c in channels], "data": data, "error": error,
               "fitted": (np.isfinite(data) & np.isfinite(error) & (error > 0)).tolist()}
        components = {em: np.asarray(per_cam[name], dtype=float) for em, per_cam in synth.items() if name in per_cam}
        if components:
            cam["model"] = np.sum(list(components.values()), axis=0)
            cam["model_components"] = components
        cameras[name] = cam
    out["cameras"] = cameras

    rad = best.get("radiation_distribution")
    if rad is not None:
        out["tpf"] = rad.get("toroidal_peaking_factor")
        out["quantities"] = rad.get("quantities", {})
        out["toroidal_profile"] = {
            "phi": rad.get("phi"),
            "power": rad.get("total_power"),
            "peak_phi": rad.get("peak_emission"),
            "components": {k: {"power": v.get("total_power"), "scale": v.get("multiplication_factor")}
                           for k, v in rad.items() if isinstance(v, dict) and "multiplication_factor" in v},
        }
    if best.get("ranges") is not None:
        out["ranges"] = best["ranges"]
    parametric = _parametric(best)
    if parametric is not None:
        out["parametric"] = parametric
    return out


def summary(emis) -> dict:
    """Every fitted time of a run as plain values, with what identifies the run."""
    info = emis.info or {}
    times = sorted(getattr(emis, "bestFits", {}) or {})
    return to_plain({
        "format": FORMAT,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "tokamak": info.get("tokamakName"),
        "shot": info.get("shot"),
        "run_config": getattr(emis, "run_config_path", None),
        "settings": {k: info.get(k) for k in ("solver", "scale_def", "error_definition", "error_floor", "error_relative", "fit_method", "dt",
                                              "injectionLocation", "injector", "enable_dphi_scaling", "vary_peak_rad_location", "toroidal_floor")},
        "angles": "radians (phi, peak_phi); degrees where a name ends in _deg",
        "times": times,
        "results": [time_summary(emis, t) for t in times],
    })


def write_summary(emis, path) -> Path:
    """Write summary(emis) to path as strict JSON (under a temporary name, then renamed). Returns the path."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary(emis), f, indent=1, allow_nan=False)
    os.replace(tmp, path)
    return path


def load_summary(path) -> dict:
    """Read a file written by write_summary."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
