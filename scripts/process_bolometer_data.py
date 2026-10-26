#!/usr/bin/env python
"""
process_bolometer_data.py -- prepare bolometer measurement files for Emis3D.

Reads measurement files in Emis3D's format (TIME, DATA_CALIBRATED as
channels x time, channelOrder), and for each one:
  1. subtracts each channel's offset              (offset: a quiet window, mean or median)
  2. crops to a time window                       (time_window)
  3. filters the traces                           (filter: median, savgol, moving_average, lowpass,
                                                   window filters, or edge-preserving tv / wavelet)
  4. removes spikes                               (spikes: replaced by a running median)
  5. scales                                       (scale)
  6. detects bad channels                         (bad_channels: manual, zero, nan, flat, negative,
                                                   spiky, saturated, and svd: inconsistent with the
                                                   time structure the other channels share)
     and optionally removes common-mode pickup    (common_mode)
  6b. estimates per-channel errors                (errors: noise plus the filtering residual, saved as
                                                   DATA_CALIBRATED_ERROR)
  7. writes the processed file in the same format, with the bad channels zeroed
     (Emis3D treats a channel that is zero for the whole shot as bad), a report
     with the reasons and a ready-to-paste run-config block, and plots of every
     channel (raw and processed) plus an overview.

Everything is set in a YAML config (see process_bolometer_data_example.yaml
beside this script); paths in it are relative to the config file. Nothing
here depends on the emis3d package, so it runs anywhere with numpy, scipy,
h5py, yaml and matplotlib.

This is the machine-independent counterpart of the DIII-D saver
(tools/DIII-D/sxrSavers/grabSXRData.py), which fetches SXR data from MDSplus
and does the same steps for one array; that saver writes Emis3D-format
files, and this script can then be run on them (or on any machine's files).

Usage:
  python scripts/process_bolometer_data.py CONFIG [--show] [--dry-run] [--no-plots]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

DEFAULTS = {
    "time_window": None,  # [t0, t1]: keep only this interval
    "offset": {"window": None, "method": "median"},  # a window with no plasma radiation; per-channel offset subtracted
    "filter": {"type": "none", "window": 11, "order": 2, "cutoff_hz": None, "filter_order": 4,  # type also: hanning, hamming, blackman
               "tv_noise_factor": 2.0, "tv_weight": None,  # tv: weight = factor x the channel's noise, or an absolute weight
               "wavelet": "db4", "wavelet_level": None, "wavelet_threshold_factor": 3.0},  # wavelet (needs PyWavelets)
    "common_mode": {"remove": False, "components": 1},  # project out pickup shared by all channels (from the quiet window)
    "errors": {"save": True, "residual_window": 21},  # per-channel error traces: noise plus the running filtering residual
    "spikes": {"remove": False, "threshold": 8.0, "window": 21},  # |x - running median| > threshold * MAD
    "scale": 1.0,
    "invert_channels": [],  # channels whose sign is flipped (wired backwards)
    "resample": None,  # {"dt": seconds}: interpolate onto a uniform time base (needed for lowpass)
    "bad_channels": {
        "manual": [],
        "zero_fraction": 0.99,  # bad if this fraction of samples is exactly zero
        "nan_fraction": 0.05,  # bad if this fraction is not finite
        "flat_fraction": 0.01,  # bad if its std is below this fraction of the median channel std
        "negative_fraction": 0.2,  # bad if this fraction of samples is below -3 x its noise (after the offset)
        "spiky_fraction": 0.05,  # bad if this fraction of samples are spikes
        "svd": {"enabled": False, "components": 3, "residual_factor": 4.0},  # bad if its residual from the shared structure exceeds factor x noise
        "saturated_fraction": 0.02,  # bad if this fraction of samples sit at the channel's maximum (a plateau)
        "action": "zero",  # zero | nan | keep: what to do with bad channels in the output
    },
    "plots": {"save": True, "per_channel": True, "overview": True, "columns": 6, "time_window": None},
}


# ---------------------------------------------------------------------------
# Config and files
# ---------------------------------------------------------------------------


def deep_merge(base: dict, over: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text()) or {}
    if "files" not in cfg or not cfg["files"]:
        raise ValueError(f"{path}: 'files' must list the measurement files to process (input, output, camera)")
    known = {"input_dir", "output_dir", "files", *DEFAULTS}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise ValueError(f"{path}: unknown setting(s) {unknown}; known: {sorted(known)}")
    return cfg, path.parent


def read_measurement(path) -> dict:
    with h5py.File(path, "r") as f:
        data = {k: f[k][()] for k in f.keys() if isinstance(f[k], h5py.Dataset)}
        attrs = dict(f.attrs)
    for key in ("TIME", "DATA_CALIBRATED", "channelOrder"):
        if key not in data:
            raise ValueError(f"{path}: no dataset {key!r} (Emis3D format needs TIME, DATA_CALIBRATED, channelOrder)")
    data["channelOrder"] = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in data["channelOrder"]]
    data["TIME"] = np.asarray(data["TIME"], dtype=float).ravel()
    data["DATA_CALIBRATED"] = np.asarray(data["DATA_CALIBRATED"], dtype=float)
    if data["DATA_CALIBRATED"].shape != (len(data["channelOrder"]), data["TIME"].size):
        raise ValueError(f"{path}: DATA_CALIBRATED is {data['DATA_CALIBRATED'].shape}, expected (channels {len(data['channelOrder'])}, times {data['TIME'].size})")
    data["_attrs"] = attrs
    return data


def write_measurement(path, data: dict, processing: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            if k.startswith("_") or k in ("channelOrder", "TIME", "DATA_CALIBRATED"):
                continue
            f.create_dataset(k, data=v)  # other datasets, carried over unchanged
        f.create_dataset("TIME", data=data["TIME"])
        f.create_dataset("DATA_CALIBRATED", data=data["DATA_CALIBRATED"])
        f.create_dataset("channelOrder", data=[c.encode("utf-8") for c in data["channelOrder"]])
        for k, v in data.get("_attrs", {}).items():
            f.attrs[k] = v
        f.attrs["processing"] = json.dumps(processing, default=str)


# ---------------------------------------------------------------------------
# Processing steps
# ---------------------------------------------------------------------------


def crop(time, sig, window):
    if not window:
        return time, sig
    keep = (time >= window[0]) & (time <= window[1])
    if keep.sum() < 2:
        raise ValueError(f"time_window {window} leaves fewer than 2 samples (data spans {time[0]:g}..{time[-1]:g})")
    return time[keep], sig[:, keep]


def resample(time, sig, step):
    new_t = np.arange(time[0], time[-1] + step / 2, step)
    return new_t, np.vstack([np.interp(new_t, time, row) for row in sig])


def subtract_offset(time, sig, settings) -> np.ndarray:
    """Per-channel offset from a quiet window: returns the offsets (zeros if no window)."""
    window = settings.get("window")
    if not window:
        return np.zeros(sig.shape[0])
    sel = (time >= window[0]) & (time <= window[1])
    if sel.sum() < 2:
        raise ValueError(f"offset window {window} contains fewer than 2 samples (data spans {time[0]:g}..{time[-1]:g})")
    reducer = np.nanmedian if settings.get("method", "mean") == "median" else np.nanmean
    return np.asarray(reducer(sig[:, sel], axis=1), dtype=float)


def noise_level(time, sig, window) -> np.ndarray:
    """
    Per-channel noise: the robust spread (MAD-based std) of successive
    differences in the offset window (else the whole trace), divided by
    sqrt(2). Differences ignore slow structure (a drifting baseline, mains
    pickup), so this is the sample-to-sample noise.
    """
    sel = (time >= window[0]) & (time <= window[1]) if window else np.ones(time.size, bool)
    block = sig[:, sel] if sel.sum() > 2 else sig
    d = np.diff(block, axis=1)
    mad = np.nanmedian(np.abs(d - np.nanmedian(d, axis=1, keepdims=True)), axis=1)
    return 1.4826 * mad / np.sqrt(2.0)


def tv_denoise_1d(y: np.ndarray, lam: float) -> np.ndarray:
    """
    Total-variation denoising of one trace, min 0.5||x - y||^2 + lam * sum|x_{k+1} - x_k|,
    by Condat's direct algorithm (IEEE SPL 2013): exact, O(N), keeps edges.
    """
    y = np.asarray(y, dtype=float)
    N = y.size
    x = np.empty(N)
    if N == 0:
        return x
    if lam <= 0:
        return y.copy()
    k = k0 = kminus = kplus = 0
    vmin, vmax = y[0] - lam, y[0] + lam
    umin, umax = lam, -lam
    while True:
        if k == N - 1:
            if umin < 0:
                x[k0:kminus + 1] = vmin
                k = k0 = kminus = kminus + 1
                vmin = y[k]
                umin = lam
                umax = y[k] + lam - vmax
            elif umax > 0:
                x[k0:kplus + 1] = vmax
                k = k0 = kplus = kplus + 1
                vmax = y[k]
                umax = -lam
                umin = y[k] - lam - vmin
            else:
                x[k0:N] = vmin + umin / (k - k0 + 1)
                return x
            if k == N - 1:
                x[k] = vmin + umin if umin < 0 else (vmax + umax if umax > 0 else vmin + umin)
                return x
            continue
        if y[k + 1] + umin < vmin - lam:  # a negative jump ends the segment
            x[k0:kminus + 1] = vmin
            k = k0 = kminus = kplus = kminus + 1
            vmin, vmax = y[k], y[k] + 2 * lam
            umin, umax = lam, -lam
        elif y[k + 1] + umax > vmax + lam:  # a positive jump ends the segment
            x[k0:kplus + 1] = vmax
            k = k0 = kminus = kplus = kplus + 1
            vmin, vmax = y[k] - 2 * lam, y[k]
            umin, umax = lam, -lam
        else:
            k += 1
            umin += y[k] - vmin
            umax += y[k] - vmax
            if umin >= lam:
                vmin += (umin - lam) / (k - k0 + 1)
                umin = lam
                kminus = k
            if umax <= -lam:
                vmax += (umax + lam) / (k - k0 + 1)
                umax = -lam
                kplus = k


def wavelet_denoise_1d(y: np.ndarray, wavelet: str, level, threshold_factor: float) -> np.ndarray:
    """Soft wavelet thresholding at threshold_factor x the noise (from the finest detail level); needs PyWavelets."""
    try:
        import pywt
    except ImportError as e:  # an optional dependency
        raise ValueError("filter type 'wavelet' needs the PyWavelets package: pip install PyWavelets") from e
    coeffs = pywt.wavedec(y, wavelet, level=level, mode="symmetric")
    sigma = 1.4826 * np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) if coeffs[-1].size else 0.0
    thr = threshold_factor * sigma
    coeffs = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    return np.asarray(pywt.waverec(coeffs, wavelet, mode="symmetric"))[: y.size]


def apply_filter(time, sig, settings, noise=None) -> np.ndarray:
    from scipy.ndimage import median_filter, uniform_filter1d
    from scipy.signal import butter, filtfilt, savgol_filter

    kind = str(settings.get("type", "none")).lower()
    w = int(settings.get("window", 11))
    if kind == "none":
        return sig
    if kind == "tv":  # edge-preserving: noise goes, steps and spikes of signal stay
        weight = settings.get("tv_weight")
        out = np.empty_like(sig)
        for i, row in enumerate(sig):
            lam = float(weight) if weight else float(settings.get("tv_noise_factor", 2.0)) * (float(noise[i]) if noise is not None else np.std(row))
            out[i] = tv_denoise_1d(row, lam)
        return out
    if kind == "wavelet":
        return np.vstack([wavelet_denoise_1d(row, str(settings.get("wavelet", "db4")), settings.get("wavelet_level"),
                                             float(settings.get("wavelet_threshold_factor", 3.0))) for row in sig])
    if kind == "median":
        return median_filter(sig, size=(1, w | 1), mode="nearest")
    if kind == "moving_average":
        return uniform_filter1d(sig, size=w, axis=1, mode="nearest")
    if kind in ("hanning", "hamming", "blackman", "bartlett"):  # the DIII-D saver's window filters
        win = getattr(np, kind)(w)
        win = win / win.sum()
        pad = w // 2
        padded = np.pad(sig, ((0, 0), (pad, pad)), mode="edge")
        return np.vstack([np.convolve(row, win, mode="valid")[: sig.shape[1]] for row in padded])
    if kind == "savgol":
        w |= 1
        return savgol_filter(sig, w, int(settings.get("order", 2)), axis=1, mode="interp")
    if kind == "lowpass":
        dts = np.diff(time)
        if dts.size and (dts.max() - dts.min()) > 1e-6 * dts.mean():
            raise ValueError("lowpass needs a uniform time base: set resample: {dt: ...}")
        fs = 1.0 / float(dts.mean())
        cutoff = settings.get("cutoff_hz")
        if not cutoff:
            raise ValueError("filter.cutoff_hz is required for a lowpass filter")
        b, a = butter(int(settings.get("filter_order", 4)), float(cutoff) / (0.5 * fs))
        return filtfilt(b, a, sig, axis=1)
    raise ValueError(f"unknown filter type {kind!r}: none, median, moving_average, savgol, lowpass, tv, wavelet, hanning, hamming, blackman or bartlett")


def find_spikes(sig, threshold: float, window: int) -> np.ndarray:
    """Boolean mask of samples far from a running median (per channel, MAD-scaled)."""
    from scipy.ndimage import median_filter

    w = int(window) | 1
    base = median_filter(sig, size=(1, w), mode="nearest")
    dev = sig - base
    mad = np.nanmedian(np.abs(dev), axis=1, keepdims=True)
    scale = np.where(mad > 0, 1.4826 * mad, np.nanstd(dev, axis=1, keepdims=True) + 1e-300)
    return np.abs(dev) > float(threshold) * scale


def svd_residuals(sig: np.ndarray, components: int) -> np.ndarray:
    """
    Each channel's RMS residual from the time structure the *other* channels
    share: the leading right singular vectors (time patterns) of the other
    channels, each scaled by its peak so bright ones do not dominate, with
    the tested channel left out (otherwise a component is spent on it).
    """
    finite = np.isfinite(sig)
    X = np.where(finite, sig, 0.0)
    scale = np.maximum(np.nanmax(np.abs(X), axis=1), 1e-300)
    Xn = X / scale[:, None]
    n = Xn.shape[0]
    out = np.zeros(n)
    for i in range(n):
        others = np.delete(Xn, i, axis=0)
        _, S, Vt = np.linalg.svd(others, full_matrices=False)
        k = int(min(max(components, 1), S.size))
        basis = Vt[:k]  # shared time patterns (orthonormal)
        approx = (Xn[i] @ basis.T) @ basis
        out[i] = np.sqrt(np.mean((Xn[i] - approx) ** 2)) * scale[i]
    return out


def remove_common_mode(time, sig, window, components: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Project out pickup shared by the channels: the leading SVD components of
    the quiet window (no plasma), applied to the whole trace. Returns the
    cleaned signal and the removed part.
    """
    sel = (time >= window[0]) & (time <= window[1]) if window else np.ones(time.size, bool)
    quiet = np.where(np.isfinite(sig[:, sel]), sig[:, sel], 0.0)
    quiet = quiet - np.median(quiet, axis=1, keepdims=True)
    # each channel in units of its own noise, outliers (spikes) clipped, so no single channel dominates
    scale = np.maximum(1.4826 * np.median(np.abs(quiet), axis=1), 1e-300)
    Q = np.clip(quiet / scale[:, None], -5.0, 5.0)
    U, S, _ = np.linalg.svd(Q, full_matrices=False)
    k = int(min(max(components, 1), S.size))
    pattern = U[:, :k]  # how the pickup is shared across channels (in noise units)
    scaled = np.where(np.isfinite(sig), sig, 0.0) / scale[:, None]
    removed = (pattern @ (pattern.T @ scaled)) * scale[:, None]
    return sig - removed, removed


def detect_bad(time, sig, noise, channels, settings, spike_mask, unfiltered=None) -> dict:
    """
    {channel: [reasons]} for the channels that fail any criterion. Zeros and
    saturation are judged on the unfiltered signal (smoothing rounds a
    plateau's edges and hides exact values); flatness and sign on the processed one.
    """
    bad: dict = {}
    exact = sig if unfiltered is None else unfiltered

    def flag(i, reason):
        bad.setdefault(channels[i], []).append(reason)

    n = sig.shape[1]
    for ch in settings.get("manual") or []:
        if ch not in channels:
            raise ValueError(f"bad_channels.manual names {ch!r}, which is not in the file ({channels[:3]}...)")
        flag(channels.index(ch), "listed in bad_channels.manual")
    finite = np.isfinite(sig)
    zero_frac = np.mean(exact == 0, axis=1)
    nan_frac = 1.0 - np.mean(finite, axis=1)
    std = np.nanstd(np.where(finite, sig, np.nan), axis=1)
    ref_std = np.nanmedian(std[std > 0]) if np.any(std > 0) else 0.0
    for i in range(sig.shape[0]):
        if zero_frac[i] >= settings["zero_fraction"]:
            flag(i, f"zero for {100 * zero_frac[i]:.0f}% of the samples")
        if nan_frac[i] >= settings["nan_fraction"]:
            flag(i, f"not finite for {100 * nan_frac[i]:.0f}% of the samples")
        elif ref_std > 0 and std[i] < settings["flat_fraction"] * ref_std and zero_frac[i] < settings["zero_fraction"]:
            flag(i, f"flat: std {std[i]:.3g} is below {settings['flat_fraction']:g} x the median channel std {ref_std:.3g}")
        neg = np.mean(sig[i] < -3.0 * max(noise[i], 1e-300)) if noise[i] > 0 else 0.0
        if neg >= settings["negative_fraction"]:
            flag(i, f"negative (below -3 x its noise) for {100 * neg:.0f}% of the samples")
        spiky = float(np.mean(spike_mask[i])) if spike_mask is not None else 0.0
        if spiky >= settings["spiky_fraction"]:
            flag(i, f"spiky: {100 * spiky:.0f}% of the samples are spikes")
        if n > 10 and np.isfinite(exact[i]).any():
            peak = np.nanmax(exact[i])
            sat = np.mean(exact[i] >= peak - 1e-9 * abs(peak)) if peak > 0 else 0.0
            if sat >= settings["saturated_fraction"] and std[i] > 0:
                flag(i, f"saturated: {100 * sat:.0f}% of the samples sit at the maximum {peak:.3g}")
    svd = settings.get("svd") or {}
    if svd.get("enabled") and sig.shape[0] > 3:
        ok_rows = [i for i in range(sig.shape[0]) if channels[i] not in bad]
        if len(ok_rows) > 3:
            resid = np.full(sig.shape[0], np.nan)
            resid[ok_rows] = svd_residuals(sig[ok_rows], int(svd.get("components", 3)))
            for i in ok_rows:
                ratio = resid[i] / max(noise[i], 1e-300) if noise[i] > 0 else 0.0
                if ratio > float(svd.get("residual_factor", 4.0)):
                    flag(i, f"inconsistent with the other channels: SVD residual {ratio:.1f} x its noise")
    return bad


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_file(out_dir, name, time_raw, raw, time, proc, channels, bad, offset_window, settings, show, errors=None):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tw = settings.get("time_window")
    saved = []

    if settings.get("overview", True):
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, ch in enumerate(channels):
            style = {"color": "0.6", "ls": "--", "lw": 0.8} if ch in bad else {"lw": 0.9}
            ax.plot(time, proc[i], label=f"{ch}{' (bad)' if ch in bad else ''}", **style)
        if offset_window:
            ax.axvspan(*offset_window, color="0.9", label="offset window")
        if tw:
            ax.set_xlim(tw)
        ax.set_xlabel("time")
        ax.set_ylabel("signal (processed)")
        ax.set_title(f"{name}: {len(channels)} channels, {len(bad)} bad")
        ax.legend(fontsize=6, ncol=4, loc="upper right")
        fig.tight_layout()
        p = out_dir / f"{name}_overview.png"
        fig.savefig(p, dpi=120)
        saved.append(p)
        if not show:
            plt.close(fig)

    if settings.get("per_channel", True):
        cols = int(settings.get("columns", 6))
        rows = int(np.ceil(len(channels) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.2 * rows), sharex=True, squeeze=False)
        for i, ch in enumerate(channels):
            ax = axes[i // cols][i % cols]
            ax.plot(time_raw, raw[i], color="0.75", lw=0.7, label="raw")
            if errors is not None:
                ax.fill_between(time, proc[i] - errors[i], proc[i] + errors[i], color="C0", alpha=0.2, lw=0, label="error")
            ax.plot(time, proc[i], color="C3" if ch in bad else "C0", lw=0.9, label="processed")
            if offset_window:
                ax.axvspan(*offset_window, color="0.92")
            reasons = "; ".join(bad.get(ch, []))
            ax.set_title(f"{ch}" + (f"\nBAD: {reasons}" if reasons else ""), fontsize=7, color="C3" if ch in bad else "k")
            ax.tick_params(labelsize=6)
            if tw:
                ax.set_xlim(tw)
        for j in range(len(channels), rows * cols):
            axes[j // cols][j % cols].axis("off")
        axes[0][0].legend(fontsize=6)
        fig.suptitle(f"{name}: raw (grey) and processed (blue; red = bad)", fontsize=10)
        fig.tight_layout()
        p = out_dir / f"{name}_channels.png"
        fig.savefig(p, dpi=120)
        saved.append(p)
        if not show:
            plt.close(fig)
    if show:
        plt.show()
    return saved


# ---------------------------------------------------------------------------


def process_file(entry: dict, cfg: dict, base: Path, show: bool, dry_run: bool, plots: bool, say) -> dict:
    settings = deep_merge(DEFAULTS, {k: v for k, v in cfg.items() if k in DEFAULTS})
    settings = deep_merge(settings, {k: v for k, v in entry.items() if k in DEFAULTS})  # per-file overrides
    in_dir = base / cfg.get("input_dir", ".")
    out_dir = base / cfg.get("output_dir", "processed")
    src = (in_dir / entry["input"]) if not Path(entry["input"]).is_absolute() else Path(entry["input"])
    dst = out_dir / entry.get("output", f"{src.stem}_PROCESSED{src.suffix}")
    name = entry.get("camera", src.stem)
    say(f"{name}: {src}")

    data = read_measurement(src)
    channels = data["channelOrder"]
    time_raw, raw = data["TIME"].copy(), data["DATA_CALIBRATED"].copy()

    # offsets and noise come from the quiet window of the full trace (it usually lies before the kept interval)
    offsets = subtract_offset(time_raw, raw, settings["offset"])
    noise = noise_level(time_raw, raw - offsets[:, None], settings["offset"].get("window"))
    time, sig = crop(time_raw, raw - offsets[:, None], settings["time_window"])
    for ch in settings.get("invert_channels") or []:
        if ch not in channels:
            raise ValueError(f"invert_channels names {ch!r}, which is not in the file")
        sig[channels.index(ch)] *= -1.0
    if settings.get("resample"):
        time, sig = resample(time, sig, float(settings["resample"]["dt"]))
    unfiltered = sig.copy()
    spike_mask = find_spikes(sig, settings["spikes"]["threshold"], settings["spikes"]["window"])
    if settings["spikes"].get("remove"):
        from scipy.ndimage import median_filter

        base_sig = median_filter(sig, size=(1, int(settings["spikes"]["window"]) | 1), mode="nearest")
        sig = np.where(spike_mask, base_sig, sig)
    cm = settings.get("common_mode") or {}
    if cm.get("remove"):
        sig, _ = remove_common_mode(time, sig, settings["offset"].get("window"), int(cm.get("components", 1)))
    before_filter = sig.copy()
    sig = apply_filter(time, sig, settings["filter"], noise=noise)
    sig = sig * float(settings["scale"])
    # --- per-channel errors: the quiet-window noise plus the running RMS of what filtering removed
    from scipy.ndimage import uniform_filter1d

    resid2 = uniform_filter1d((before_filter * float(settings["scale"]) - sig) ** 2, size=int(settings["errors"].get("residual_window", 21)) | 1,
                              axis=1, mode="nearest")
    errors = np.sqrt((noise[:, None] * float(settings["scale"])) ** 2 + resid2)

    bad = detect_bad(time, sig, noise, channels, settings["bad_channels"], spike_mask, unfiltered)
    action = settings["bad_channels"].get("action", "zero")
    out_sig = sig.copy()
    for ch in bad:
        i = channels.index(ch)
        if action == "zero":
            out_sig[i] = 0.0
        elif action == "nan":
            out_sig[i] = np.nan
    for ch, reasons in bad.items():
        say(f"  bad {ch}: {'; '.join(reasons)}")
    say(f"  {len(channels)} channels, {len(bad)} bad; offsets {np.min(offsets):.3g}..{np.max(offsets):.3g}; "
        f"noise median {np.nanmedian(noise):.3g}; {time.size} samples {time[0]:g}..{time[-1]:g}")

    processing = {"source": str(src), "processed": dt.datetime.now().isoformat(timespec="seconds"), "settings": settings,
                  "offsets": dict(zip(channels, [float(o) for o in offsets], strict=True)),
                  "noise": dict(zip(channels, [float(v) for v in noise], strict=True)), "bad_channels": bad}
    if not dry_run:
        out = dict(data)
        out["TIME"], out["DATA_CALIBRATED"] = time, out_sig
        if settings["errors"].get("save", True):
            out["DATA_CALIBRATED_ERROR"] = errors
        write_measurement(dst, out, processing)
        say(f"  wrote {dst}")
    figures = []
    if plots and settings["plots"].get("save", True):
        figures = plot_file(out_dir / "plots", name, time_raw, raw, time, sig, channels, bad, settings["offset"].get("window"), settings["plots"], show,
                            errors=errors)
        for p in figures:
            say(f"  plot {p}")
    return {"camera": name, "input": str(src), "output": str(dst), "bad_channels": bad, "noise": processing["noise"],
            "offsets": processing["offsets"], "n_samples": int(time.size), "figures": [str(p) for p in figures]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="YAML config (see process_bolometer_data_example.yaml)")
    ap.add_argument("--show", action="store_true", help="display the plots")
    ap.add_argument("--dry-run", action="store_true", help="process and report, but write nothing")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    say = lambda text: print(text, flush=True)  # noqa: E731

    cfg, base = load_config(args.config)
    results = [process_file(entry, cfg, base, args.show, args.dry_run, not args.no_plots, say) for entry in cfg["files"]]

    # --- The report, with a run-config block ready to paste
    out_dir = base / cfg.get("output_dir", "processed")
    block = {"BOLOMETERS": {r["camera"]: {"dataFileName": Path(r["output"]).name, **({"badChannels": sorted(r["bad_channels"])} if r["bad_channels"] else {})}
                            for r in results}}
    report = {"processed": dt.datetime.now().isoformat(timespec="seconds"), "config": str(Path(args.config).resolve()), "files": results,
              "run_config_block": block}
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "processing_report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
        say(f"report: {out_dir / 'processing_report.yaml'}")
    say("\nrun config block (dataFileName is relative to the measurements location; adjust the folder):")
    say(yaml.safe_dump(block, sort_keys=False).rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
