# Util_emis3D.py
"""
Contains scaling and residual functions used by emis3D.

Note: All fitting is done centered around zero, dphi = phi - mu, where
mu is the injection location

Written by JLH Aug. 2025
"""

import numpy as np

from main.Util import convert_arrays_to_list
from scipy.special import i0
from lmfit import minimize


def _exp(dphi=np.ndarray(()), kappa=0.0) -> "np.ndarray":
    """Exponential function"""
    return np.exp(-1.0 * kappa * (dphi**2))


def scale_exp(A=0.0, B=0.0, dphi=np.ndarray(())) -> "np.ndarray":
    raw = _exp(dphi, B)
    normalized = raw / _exp(np.zeros(1), B)
    return A * normalized


def scale_linear(A=0.0, B=0.0, dphi=np.ndarray(())) -> "np.ndarray":
    return A * dphi + B


def scale_constant(A=0.0, dphi=np.ndarray(())) -> "np.ndarray":
    return A * np.ones(dphi.shape[0])


def _von_mises(dphi=np.ndarray(()), kappa=0.0) -> "np.ndarray":
    """Von Mises distribution, normalized"""
    return np.exp(kappa * np.cos(dphi)) / (2.0 * np.pi * i0(kappa))


def von_mises_amplitude(
    A=0.0, B=0.0, dphi=np.ndarray(()), mu=0.0, emissionName=None
) -> "np.ndarray":
    """
    Apply scaling to Von Mises distribution, while ensuring that the
    endpoints are equal.

    left : theta-mu in (-pi, 0], counterClock
    right : theta-mu in (0, pi], clockwise
    """

    # --- Normalize VM so the value at mu = 1.0
    # theta = mu, so dphi = mu - mu = 0
    raw = _von_mises(dphi, B)
    normalized = raw / _von_mises(np.zeros(1), B)
    return A * normalized


def scale_wrapper(
    a=0.0,
    b=0.0,
    phi=np.ndarray(()),
    mu=0.0,
    scale_def=None,
    emissionName=None,
    dphi=None,
    numRevolutions=1.0,
):
    """
    Wrapper for the scale function, returns scaling factor
    based off the scale_def.

    A         :: float
                 The amplitude of the scaling function
    B         :: float
                 Value used by most scaling functions
    theta     :: np.ndarray (radians)
                 Location of the bolomter
    mu        :: float (radians)
                 injection location
    scale_def ::


    Returns :: np.ndarray
    """

    # --- Find dphi
    if dphi is None:
        dphi = find_dphi(
            phi, mu, emissionName=str(emissionName), numRevolutions=numRevolutions
        )

    # --- Set a = 0 for small values of a
    if a < 0.01:
        a = 0
    if scale_def == "exponential":
        scale_ = scale_exp(a, b, dphi)
    elif scale_def == "linear":
        scale_ = scale_linear(a, b, dphi)
    elif scale_def == "constant":
        scale_ = scale_constant(a, dphi)
    elif scale_def == "von_mises":
        scale_ = von_mises_amplitude(a, b, dphi, emissionName=emissionName)
    else:
        scale_ = np.ones(phi.shape[0])

    return scale_


def find_dphi(
    phi=np.ndarray(()), mu=0.0, emissionName="", scale=True, numRevolutions=1.0
) -> "np.ndarray":
    """
    Finds the change in toroidal angle between the inputs phi and mu. Example:
    phi = 220, mu = 100, dphi = 120 or 240 depending on the emssionName input. Output is in radians.

    It will also correct the phi location for helical fits based on if the direction is
    counterClock (0, pi], or clockwise [-pi, 0). It will also scale phi
    by 0.5 since the helical distributions go 2pi if the mode is clockwise or counterClock. It
    will not scale phi for other distributions.

    phi  :  list (radians)
            List of length of the number of channels of the toroidal location of the bolometer
    mu   :  float
            Typically the injection location

    emissionName : str
        "clockwise"     -> dphi in the clockwise direction, as from looking down on the tokamak
        "counterClock"  -> dphi in the counter clockwise direction
        anything other than clockwise and counterClock   -> minimium distance between phi and mu

    Returns:
            dphi :: np.ndarray
                    Corrected phi's in radians
    """

    two_pi = 2 * np.pi

    # --- Find if we need to add 2 pi to the end result
    add_2pi = phi > two_pi

    phi = phi % two_pi
    mu = mu % two_pi

    cw = (mu - phi) % two_pi
    ccw = (phi - mu) % two_pi

    # --- Correct for values that are greater than 2pi
    ccw[add_2pi] += two_pi
    cw[add_2pi] += two_pi

    # --- Return values for helical distributions
    if "clockwise" in emissionName:
        # --- Helical distributions go a full revolution around the machine. Example:
        # counter-clock goes from the injection location back to the injection location
        # So we need to scale phi down to +/- pi for the fit, and scale it back up after
        # the fit
        #
        # We also need to account for the total number of revolutions the helical distribution
        # makes around the machine.
        if scale:
            return -1.0 * cw / (2.0 * numRevolutions)
        else:
            return -1.0 * cw
    elif "counterClock" in emissionName:
        if scale:
            return ccw / (2.0 * numRevolutions)
        else:
            return ccw

    # --- Return the shortest value, used for the elongated rings
    else:
        if sum(ccw) < sum(cw):
            return ccw
        else:
            return -cw


def residual(
    pars, data_dict, synthetic_dict, scale_def=None, boloNames=None, residual=True
):
    """
    Returns the residual for the fit

    """

    a = 0.0
    b = 0.0
    params = pars.valuesdict()
    mu = float(synthetic_dict["injectionLocation_rad"])

    # --- Find the total synthetic emission
    temp_ = {}
    data = {}

    for emissionName in synthetic_dict["emissionNames"]:
        # --- Find the number of revolutions the helical distribution makes,
        # it will return 0 for non-helical distributions
        if "clockwise" or "counterClock" in emissionName:
            numRevolutions = len(synthetic_dict["emissionNames"]) / 2.0
        else:
            numRevolutions = 0.0

        data[emissionName] = {}
        # --- Get the new scale factor for the normal runs
        if boloNames is None:
            # --- Hard-coded parameter names... not ideal
            a = params[f"a_{synthetic_dict['injectionLocation']}"]
            if "clockwise" in emissionName:
                b = params[f"b_clockwise_{synthetic_dict['injectionLocation']}"]
            elif "counterClock" in emissionName:
                b = params[f"b_counterClock_{synthetic_dict['injectionLocation']}"]
            else:
                b = params[f"b_{emissionName}_{synthetic_dict['injectionLocation']}"]

        # --- Loop over each bolometer group
        for ii in range(len(synthetic_dict[emissionName]["data"])):

            if boloNames is not None:
                a = params[f"{boloNames[ii]}"]
                b = 0.0

            phi = np.array(synthetic_dict[emissionName]["scaleFactor"][ii])

            # --- Return the scale factor
            scale_ = scale_wrapper(
                a,
                b,
                phi=phi,
                mu=mu,
                scale_def=scale_def,
                emissionName=emissionName,
                numRevolutions=numRevolutions,
            )

            synth_ = np.array(synthetic_dict[emissionName]["data"][ii])
            # synth_error = np.array(synthetic_dict[emissionName]["data_error"][ii])

            if ii not in temp_:
                temp_[ii] = np.zeros(len(scale_))

            temp_[ii] += scale_ * synth_

            data[emissionName][ii] = scale_ * synth_

    if residual and data_dict is not None:
        res = []
        # --- Loop over each bolometer group
        for ii in temp_:
            data_ = np.array(data_dict["observed"][ii].copy())
            data_error = np.array(data_dict["observed_error"][ii].copy())

            # --- Make sure there are no zeros in the error
            data_error[data_error <= 1.0e-6] = 1.0e-6

            # --- Ignore bad data points
            bad_indices = np.where(data_ <= 0)[0]
            temp_[ii][bad_indices] = data_[bad_indices]

            numerator = data_ - temp_[ii]
            # NOTE: LMFIT minimizes the sum of squares of the residuals
            # so we do not need to square the residuals here
            answer = convert_arrays_to_list(numerator)  # / data_error)
            res.extend(answer)

        return res
    else:
        return data


def runParallel(job):
    boloNames = None
    res_ = True
    fit_index, pars, data_dict, synth_dict, scale_def = job

    fit = minimize(
        residual,
        pars,
        args=(
            data_dict,
            synth_dict,
            scale_def,
            boloNames,
            res_,
        ),
        method="leastsq",
    )

    return fit_index, fit
