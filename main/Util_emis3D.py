# Util_emis3D.py
"""
Contains scaling and residual functions used by emis3D.

Note: All fitting is done centered around zero, dphi = phi - mu, where
mu is the injection location

Written by JLH Aug. 2025

TODO: Account for rev1, rev2, etc. for the helical distributions. The dphi
"""

import logging
import re
import numpy as np

from main.Util import convert_arrays_to_list
from scipy.special import i0
from lmfit import minimize

logger = logging.getLogger(__name__)


def _get_observed_phi_loc(emission_dict: dict):
    """
    Return the toroidal observation angles from a synthetic_dict emission
    entry. Newer radDists store these under 'observed_phi_loc'; older saved
    radDists used the name 'scaleFactor'.
    """
    phi_loc = emission_dict.get("observed_phi_loc", emission_dict.get("scaleFactor"))
    if phi_loc is None:
        raise KeyError(
            "synthetic_dict entry contains neither 'observed_phi_loc' nor the "
            "legacy 'scaleFactor' key"
        )
    return phi_loc


def _as_channel_segments(rows) -> list:
    """
    Normalize a per-bolometer-group list so that every channel entry is a 1D
    float array of toroidal segments. Scalars (the legacy single-plane format)
    are promoted to single-element arrays; empty channels become a single zero
    segment so downstream per-channel reductions stay well-defined.
    """
    out = []
    for chan in rows:
        arr = np.atleast_1d(np.asarray(chan, dtype=float))
        if arr.size == 0:
            arr = np.zeros(1)
        out.append(arr)
    return out


def _exp(dphi: np.ndarray, kappa: float = 0.0) -> np.ndarray:
    """Exponential function. Peak = 1.0 at dphi = 0."""
    return np.exp(-1.0 * kappa * (dphi**2))


def scale_exp(A: float, B: float, dphi: np.ndarray) -> np.ndarray:
    """Gaussian (exponential) scaling. Peak is A at dphi = 0"""
    return A * _exp(dphi, B)


def scale_linear(
    A: float,
    B: float,
    dphi: np.ndarray,
) -> np.ndarray:
    """
    Triangular profile centered at mu.

    Peak = A at dphi = 0. Should not be negative.
    """
    return np.abs(A - np.abs(B) * dphi)


def scale_constant(A: float, dphi: np.ndarray) -> np.ndarray:
    return A * np.ones(dphi.shape[0])


def scale_step(
    A: float, B: float, phi: np.ndarray, bolo_phi_locs: np.ndarray, mu: float=None) -> np.ndarray:
    """
    Inputs:
    A: float, use as the highest amplitude in the step function
    B: float, use as the lowest amplitude in the step function
    phi: Input phi locations (in radians) for every channel for every bolometer (more for toroidal bolometers)
    mu: injection location
    bolo_phi_locs: list of just the bolometer phi locations (ex, [90, 225] for JET)


    """

    A_arr1 = np.array([A, B]) # this is the un-generalized step. Ideally the function would just take
                                  # A as an array input instead of floats
    A_arr = np.concatenate((A_arr1, A_arr1, A_arr1))
    bolo_phis = np.concatenate((bolo_phi_locs, bolo_phi_locs+(2*np.pi), bolo_phi_locs-(2*np.pi)))

    amplitudes = np.zeros(shape=phi.shape)
    bolo_phis_arr = np.tile(bolo_phis, (len(phi), 1))
    phi_arr = np.tile(phi, (len(bolo_phis), 1)).transpose()
    dists = np.abs(bolo_phis_arr-phi_arr)
    dists_diffs = dists - np.min(dists, 1)[:, np.newaxis]
    nearest_bolo_phi_indxs = np.argwhere(dists_diffs <1e-9)[:,1]
    amplitudes = A_arr[nearest_bolo_phi_indxs]

    return amplitudes

def scale_wrapper(
    a: float,
    b: float,
    phi: np.ndarray,
    mu: float = 0.0,
    scale_def: str | None = None,
    emissionName: str | None = None,
    dphi: np.ndarray | None = None,
    bolo_phi_locs: np.ndarray | None = None,
):
    """
    Wrapper for the scale function; returns a scaling factor array based on
    the chosen scale_def.

    Parameters
    ----------
    a             : Amplitude of the scaling function
    b             : Shape parameter used by most scaling functions
    phi           : Toroidal location(s) of each channel (radians)
    mu            : Injection location (radians)
    scale_def     : One of 'exponential', 'linear', 'constant'
    emissionName  : Name of the emission distribution (e.g. 'clockwise')
    dphi          : Pre-computed dphi; calculated from phi/mu if None
    numRevolutions: Number of toroidal revolutions for helical distributions
    bolo_phi_locs : Toroidal locations of the bolometers (radians)
    """

    if phi is None and dphi is None:
        raise ValueError(
            "Phi or dphi array must be populated when calling scale_wrapper!"
        )

    # --- Find dphi
    if dphi is None and phi is not None:
        dphi = find_dphi(
            phi,
            mu,
            emissionName=str(emissionName),
        )

    if dphi is None:
        return

    if scale_def == "exponential":
        return scale_exp(a, b, dphi)
    elif scale_def == "linear":
        return scale_linear(a, b, dphi)
    elif scale_def == "constant":
        return scale_constant(a, dphi)
    elif scale_def == "step":
        if bolo_phi_locs is not None:
            return scale_step(A=a, B=b, phi=phi, bolo_phi_locs=bolo_phi_locs, mu=mu)
        else:
            print(
                f"Warning! bolo_phi_locs is None in the scale_wrapper definition and is required for the scale_step def"
            )
            return np.zeros(len(dphi))
    else:
        return np.ones(dphi.shape[0])


def find_dphi(
    phi: np.ndarray,
    mu: float = 0.0,
    emissionName: str = "",
) -> np.ndarray:
    """
    Finds the change in toroidal angle between phi and mu.

    Always returns the minimum distance for non-Helical distributions

    Examples
    --------
        phi = 220, mu = 100
        dphi = 120 for counterClock and non-Helical distributions
        dphi = 240 for clockwise Helical distributions

        phi = -370, mu = 30
        dphi = 40 + 360 = 400 for counterClock Helical and non-Helical distributions
        dphi = 320 + 360 = 680 for clockwise Helical distributions

        phi = 275, mu = 260
        dphi = 345 for counterClock Helical
        dphi = 15 for clockwise Helical and non-Helical distributions


    Parameters
    ----------
    phi           : Toroidal locations of the bolometers (radians)
    mu            : Injection location (radians)
    emissionName  : 'clockwise', 'counterClock', or anything else for minimum
                    angular distance
    scale         : Whether to scale helical dphi into [-pi, pi]
    numRevolutions: Total revolutions of the helical distribution

    Returns
    -------
    dphi : np.ndarray
    """

    if emissionName is None:
        logger.warning("Emission name in Util_emis3D.find_dphi is None.")
        return None

    phi = np.asarray(phi, dtype=float)

    two_pi = 2.0 * np.pi

    # Number of full revolutions represented by phi
    n_wraps = np.floor(np.abs(phi) / two_pi).astype(int)

    # exact multiples of 360 should not count the final wrap
    n_wraps = np.where(
        (np.abs(phi) > 0) & (np.mod(np.abs(phi), two_pi) == 0),
        n_wraps - 1,
        n_wraps,
    )

    wraps = n_wraps * two_pi

    phi_mod = phi % two_pi
    mu_mod = mu % two_pi

    # Positive directional distances
    ccw = (phi_mod - mu_mod) % two_pi
    cw = (mu_mod - phi_mod) % two_pi

    # Add back complete revolutions
    ccw += wraps
    cw += wraps

    # --- Account for 2nd or more revolutions around the tokamak
    rev_match = re.search(r"_rev(\d+)", emissionName)
    n_rev = int(rev_match.group(1)) if rev_match else 0
    rev_offset = n_rev * two_pi

    if "clockwise" in emissionName:
        return cw + rev_offset

    if "counterClock" in emissionName:
        return ccw + rev_offset

    return np.minimum(ccw, cw)


def loc_tag(injectionLocation) -> str:
    """Canonical tag for LMFIT parameter names ('a_240', 'b_ ... _ 240', etc.)
    Float injection locations must be int to agree with the creation"""
    return str(int(round(float(injectionLocation))))


def _emission_pairs(emissionNames) -> list:
    """
    Pair the clockwise / counterClock helical emission directions that share the
    same revolution suffix.

    Examples
    --------
        ['clockwise_rev0', 'counterClock_rev0'] -> [('clockwise_rev0',
                                                     'counterClock_rev0')]

    Returns
    -------
    list of (clockwise_name, counterClock_name) tuples.
    """
    cw, ccw = {}, {}
    for name in emissionNames:
        # 'clockwise' is not a substring of 'counterClock', so the order of
        # these checks does not cause a mis-classification.
        if "clockwise" in name:
            cw[name.replace("clockwise", "")] = name
        elif "counterClock" in name:
            ccw[name.replace("counterClock", "")] = name

    return [(cw[s], ccw[s]) for s in cw if s in ccw]


def helical_endpoint_penalty(
    pars,
    synthetic_dict,
    scale_def: str | None = None,
    weight: float = 0.0,
) -> list:
    """
    Soft constraint tying paired clockwise / counterClock helical distributions
    together at the helix *endpoint*.

    The peak (dphi = 0) of the two directions is already tied together because
    both share the universal amplitude ``a_<tag>``. What is *not* tied is the
    endpoint, i.e. the point reached after one full toroidal transit, which sits
    back at the injection toroidal angle (phi = injectionLocation mod 2*pi).
    This function adds that missing condition.

    For a direction d the endpoint radiated power is::

        P_end_d = scale_d(dphi_end) * P_pol_d(phi = mu mod 2*pi) * scaleSynth

    where ``dphi_end = 2*pi * numTransists`` is the winding distance back to the
    injection toroidal angle.

    ``(data - model) / error`` residuals that LMFIT already minimizes::

        weight * (P_end_cw - P_end_ccw) / (P_peak + eps)

        P_peak = a * scaleSynth * 0.5*(P_pol_cw(mu) + P_pol_ccw(mu))

    Normalising by the peak keeps it unitless.

    Returns an empty list (no constraint) when:
      * ``weight == 0``,
      * no clockwise/counterClock pair is present (non-helical distribution),
      * the radiated-power profile (``P_pol`` / ``phi_array``) was not threaded
        into ``synthetic_dict`` (e.g. older saved radDists), or
      * the shared amplitude ``a_<tag>`` is absent (cross-calibration fits).

    Notes
    -----
    The radiated-power profile is stored on the radDist under
    ``data['toroidalRadiatedPower'][emissionName]`` with keys ``'phi_array'``
    and ``'P_pol'`` (NOT ``'phi'``); ``_build_synthetic_dict`` copies those into
    ``synthetic_dict[emissionName]``.
    """

    if not weight:
        return []

    pairs = _emission_pairs(synthetic_dict.get("emissionNames", []))
    if not pairs:
        return []

    params = pars.valuesdict()
    tag = loc_tag(synthetic_dict["injectionLocation"])

    a_name = f"a_{tag}"
    if a_name not in params:
        # cross-calibration mode has no shared amplitude; nothing to tie together
        return []
    a = params[a_name]

    # Injection / peak location (rad)
    mu = float(synthetic_dict["injectionLocation_rad"])
    if "peak_rad_loc" in params:
        mu = float(params["peak_rad_loc"])

    eps = 1.0e-30
    penalties = []

    for cw_name, ccw_name in pairs:
        endpoint_power = {}
        peak_Ppol = {}
        usable = True

        for name in (cw_name, ccw_name):
            sub = synthetic_dict.get(name, {})
            phi_arr = np.asarray(sub.get("phi_array", []), dtype=float)
            P_pol = np.asarray(sub.get("P_pol", []), dtype=float)
            if phi_arr.size == 0 or P_pol.size == 0:
                usable = False
                break

            b = params.get(f"b_{name}_{tag}", 0.0)

            # Scale evaluated at the endpoint winding distance (NOT at phi = mu,
            # which would return the dphi = 0 peak).
            scale_end = scale_wrapper(
                a,
                b,
                phi=phi_arr,
                mu=mu,
                scale_def=scale_def,
                emissionName=name,
            )

            if scale_end is None:
                usable = False
                break

            # P_pol sampled at the injection toroidal location (mod 2*pi)
            idx = int(np.argmin(np.abs(phi_arr - mu)))
            scaleSynth = float(sub.get("scaleSynth", 1.0))

            endpoint_power[name] = scale_end[idx] * P_pol[idx] * scaleSynth
            peak_Ppol[name] = P_pol[idx] * scaleSynth

        if not usable:
            continue

        p_cw = endpoint_power[cw_name]
        p_ccw = endpoint_power[ccw_name]

        # Peak power shared reference (dphi = 0 amplitude is 'a'); keeps the
        # penalty bounded and ->0 when both endpoints have decayed away.
        p_peak = a * 0.5 * (peak_Ppol[cw_name] + peak_Ppol[ccw_name])
        penalties.append(weight * (p_cw - p_ccw) / (p_peak + eps))

    return penalties


def residual(
    pars,
    data_dict,
    synthetic_dict,
    scale_def: str | None = None,
    boloNames=None,
    residual: bool = True,
    helical_endpoint_weight: float = 0.0,
):
    """
    Computes the residual (or the scaled synthetic data) for the minimizer.

    When ``residual=True`` returns a flat list of (data - model) values for
    each bolometer channel (LMFIT squares these internally).
    When ``residual=False`` returns a dict of scaled synthetic arrays keyed
    by emissionName → bolometer-group index. This is used to make synthetic data
    in Emis3D._post_process_fit_arrangement()

    ``helical_endpoint_weight`` (only used when ``residual=True``) controls the
    soft constraint that ties paired clockwise / counterClock helical
    distributions together at the helix endpoint; see
    :func:`helical_endpoint_penalty`. A weight of 0 disables the constraint.
    """

    a = 0.0
    b = 0.0
    params = pars.valuesdict()

    # --- Check to see if the peak radiation location can be varied
    mu = float(synthetic_dict["injectionLocation_rad"])
    if "peak_rad_loc" in params:
        mu = float(params["peak_rad_loc"])

    # --- Find the total synthetic emission
    temp_ = {}  # accumulated model signal per bolometer group
    data = {}  # per-emission scaled synthetic (returned when residual=False)

    # --- Find the bolometer phi locations, combine to one large list
    bolo_phis = []
    for emissionName in synthetic_dict["emissionNames"]:
        phi_loc = _get_observed_phi_loc(synthetic_dict[emissionName])
        for group in phi_loc:
            for chan in group:
                bolo_phis.extend(np.atleast_1d(chan).tolist())

    # Remove duplicates
    bolo_phis = np.asarray(list(set(bolo_phis)))

    for emissionName in synthetic_dict["emissionNames"]:

        data[emissionName] = {}
        tag = loc_tag(synthetic_dict["injectionLocation"])
        # --- Get the new scale factor for the normal runs
        if boloNames is None:
            a = params[f"a_{tag}"]
            b = params[f"b_{emissionName}_{tag}"]

        # --- Loop over each bolometer group
        for ii in range(len(synthetic_dict[emissionName]["data"])):

            # --- Different parameter names when doing the cross-calibration
            if boloNames is not None:
                a = params[f"{boloNames[ii]}"]
                b = 0.0

            # phi = np.array(synthetic_dict[emissionName]["scaleFactor"][ii])

            # --- Per-channel lists of toroidal observation segments. Each
            # channel holds one phi and one synthetic value per segment
            # (single-element lists for poloidal-fan cameras).

            phi_segs = _as_channel_segments(
                _get_observed_phi_loc(synthetic_dict[emissionName])[ii]
            )
            synth_segs = _as_channel_segments(synthetic_dict[emissionName]["data"][ii])

            # --- Check to make sure lists are equal
            if len(phi_segs) != len(synth_segs):
                raise RuntimeError(
                    f"observed_phi_loc and data have different channel counts "
                    f"({len(phi_segs)} vs {len(synth_segs)}) for "
                    f"'{emissionName}', bolometer group {ii}"
                )

            for jj, (p_, s_) in enumerate(zip(phi_segs, synth_segs)):
                if len(p_) != len(s_):
                    raise RuntimeError(
                        f"Channel {jj} of '{emissionName}', bolometer group "
                        f"{ii} has {len(p_)} observed_phi_loc segments but "
                        f"{len(s_)} data segments"
                    )

            phi_flat = np.concatenate(phi_segs)
            synth_flat = np.concatenate(synth_segs)

            # --- Return the scale factor for every segment
            scale_ = scale_wrapper(
                a,
                b,
                phi=phi_flat,
                mu=mu,
                scale_def=scale_def,
                emissionName=emissionName,
                bolo_phi_locs=bolo_phis,
            )

            if scale_ is None:
                raise RuntimeError("Scale_ returned None for some reason!?!")

            # --- Sum the scaled segments within each channel so the model is
            # one value per channel, matching the measured signal
            seg_counts = [len(p) for p in phi_segs]
            starts = np.cumsum([0] + seg_counts[:-1])
            model_ = np.add.reduceat(scale_ * synth_flat, starts)

            if ii not in temp_:
                temp_[ii] = np.zeros(len(model_))

            temp_[ii] += model_
            data[emissionName][ii] = model_

    if residual and data_dict is not None:
        res = []
        # --- Loop over each bolometer group
        for ii in temp_:
            data_ = np.array(data_dict["observed"][ii].copy())
            data_error = np.array(data_dict["observed_error"][ii].copy())

            # --- Make sure there are no zeros in the error
            data_error[data_error <= 1.0e-6] = 1.0e-6

            # --- Ignore channels with non-positive observed values
            bad_indices = np.where(data_ <= 0)[0]
            # --- Set the temp_ values equal to the bad data values so the residual is zero
            temp_[ii][bad_indices] = data_[bad_indices]

            # LMFIT minimizes the sum of squares, so we return the raw residual
            numerator = np.abs(data_ - temp_[ii])
            res.extend(convert_arrays_to_list(numerator / data_error))

        # --- Soft constraint: tie paired clockwise / counterClock helical
        # distributions together at the helix endpoint (phi = mu mod 2*pi).
        # Returns [] for non-helical fits or when weight == 0.
        res.extend(
            helical_endpoint_penalty(
                pars,
                synthetic_dict,
                scale_def=scale_def,
                weight=helical_endpoint_weight,
            )
        )

        return res
    else:
        return data


def runParallel(job):
    """Thin wrapper around residual minimization for use with ProcessPoolExecutor."""
    boloNames = None
    res_ = True
    fit_index, pars, data_dict, synth_dict, scale_def, helical_endpoint_weight = job

    fit = minimize(
        residual,
        pars,
        args=(
            data_dict,
            synth_dict,
            scale_def,
            boloNames,
            res_,
            helical_endpoint_weight,
        ),
        method="leastsq",
    )

    return fit_index, fit


def signal_error(
    type: str,
    signal: np.ndarray,
    max_signal: float,
    scale_factor: float = 1.0,
):
    """Wrapper for the error function"""

    type_ = type.upper()

    if type_ == "EXPONENTIAL":
        return error_exponential(signal, max_signal, scale_factor)
    elif type_ == "INVERSE":
        return error_inverse(signal, max_signal, scale_factor)
    elif type_ == "INVERSE SQUARE":
        return error_inv_sqrt(signal, max_signal, scale_factor)
    elif type_ == "CONSTANT":
        return error_constant(signal, max_signal)


def error_constant(
    signal: np.ndarray,
    max_signal: float,
    _floor: float = 1.0e-12,
    fraction: float = 0.1,  # 10%
) -> np.ndarray:
    """Returns fraction * max_signal for each signal"""
    signal = np.asarray(signal)
    signal = np.clip(signal, _floor, None)

    return fraction * max_signal / signal


def error_exponential(
    signal: np.ndarray,
    max_signal: float,
    scale_factor: float = 1.0,
    decay: float = 4.0,
) -> np.ndarray:
    """
    Exponential decay error model.

    error ≈ scale_factor at signal = 0
    error ≈ 0            at signal = max_signal
    """
    signal = np.clip(signal, 0.0, None)
    return scale_factor * np.exp(-decay * signal / max_signal)


def error_inverse(
    signal: np.ndarray,
    max_signal: float,
    scale_factor: float = 1.0,
    _floor: float = 1.0e-12,
) -> np.ndarray:
    """
    Hyperbolic error model: error = scale_factor * (max_signal / signal).
    """
    signal = np.clip(signal, _floor, None)
    return scale_factor * (max_signal / signal)


def error_inv_sqrt(
    signal: np.ndarray,
    max_signal: float,
    scale_factor: float = 1.0,
    _floor: float = 1.0e-12,
) -> np.ndarray:
    """
    Inverse square-root (Poisson / shot-noise) error model.

    error = scale_factor * sqrt(max_signal / signal)
    error = 1.0 at peak signal.
    """
    signal = np.clip(signal, _floor, None)
    return scale_factor * np.sqrt(max_signal / signal)


def print_intro() -> None:

    print("""

    ███████╗███╗   ███╗██╗███████╗██████╗ ██████╗ 
    ██╔════╝████╗ ████║██║██╔════╝╚════██╗██╔══██╗
    █████╗  ██╔████╔██║██║███████╗ █████╔╝██║  ██║
    ██╔══╝  ██║╚██╔╝██║██║╚════██║ ╚═══██╗██║  ██║
    ███████╗██║ ╚═╝ ██║██║███████║██████╔╝██████╔╝
    ╚══════╝╚═╝     ╚═╝╚═╝╚══════╝╚═════╝ ╚═════╝
    """)
