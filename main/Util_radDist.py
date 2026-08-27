# Util_RadDist.py
"""
File contains defintions used while creating radDists

Written by JLH Aug. 2025

TODO:
1. Should we normalize the bivariate distributions to 2 pi? Since we integrate
over the whole tokamak while calculating the radiated power?
"""

import logging
import os

os.environ["KMP_WARNINGS"] = "FALSE"  # suppress the numba deprecation warning
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # the modern equivalent setting

import random
import numpy as np
import main.radDist as radDist
import main.Util as Util
import numba as nb
import matplotlib.path
from main.Tokamak import Tokamak

logger = logging.getLogger(__name__)


def radDist_ElongatedRing_parallel(
    args: tuple, return_result: bool = False
):  # -> None | radDist.ElongatedRing:
    """
    Worker function for parallel computation of ElongatedRing radial distribution.

    Designed to be called via multiprocessing.Pool.map, which requires a single
    argument. The args tuple is unpacked internally.

    Parameters
    ----------
    args : tuple of (rz_array, config)
        rz_array : array-like of length 2 — (R, z) start coordinates in metres.
        config   : configuration object passed to radDist.ElongatedRing.
    """
    rzArray, config = args
    elongatedRing = radDist.ElongatedRing(
        startR=rzArray[0], startZ=rzArray[1], config=config
    )
    elongatedRing.build()

    logger.info(
        f"DONE with elongatedRing radDist, R = {rzArray[0]:.2f}, z = {rzArray[1]:.2f}"
    )
    print(
        f"DONE with elongatedRing radDist, R = {rzArray[0]:.2f}, z = {rzArray[1]:.2f}"
    )

    if return_result:
        return elongatedRing


def radDist_Helical_parallel(
    args: tuple, return_result: bool = False
):  # -> None | radDist.Helical:
    """
    Worker function for parallel computation of Helical radial distribution.

    Designed to be called via multiprocessing.Pool.map, which requires a single
    argument. The args tuple is unpacked internally.

    Parameters
    ----------
    args : tuple of (rz_array, config)
        rz_array : array-like of length 2 — (R, z) start coordinates in metres.
        config   : configuration object passed to radDist.Helical.
    """

    rzArray, config = args
    helical = radDist.Helical(startR=rzArray[0], startZ=rzArray[1], config=config)
    helical.build()

    logger.info(
        f"DONE with elongatedRing radDist, R = {rzArray[0]:.2f}, z = {rzArray[1]:.2f}"
    )
    print(
        f"DONE with elongatedRing radDist, R = {rzArray[0]:.2f}, z = {rzArray[1]:.2f}"
    )

    if return_result:
        return helical


def radDist_HelicalRing_parallel(
    args: tuple, return_result: bool = False
):  # -> None | radDist.HelicalRing:
    """
    Worker function for parallel computation of Helical radial distribution.

    Designed to be called via multiprocessing.Pool.map, which requires a single
    argument. The args tuple is unpacked internally.

    Parameters
    ----------
    args : tuple of (rz_array, config)
        rz_array : array-like of length 2 — (R, z) start coordinates in metres.
        config   : configuration object passed to radDist.Helical.
    """

    rzArray, config = args
    helical = radDist.HelicalRing(startR=rzArray[0], startZ=rzArray[1], config=config)
    helical.build()

    logger.info(
        f"DONE with elongatedRing radDist, R = {rzArray[0]:.2f}, z = {rzArray[1]:.2f}"
    )
    print(
        f"DONE with elongatedRing radDist, R = {rzArray[0]:.2f}, z = {rzArray[1]:.2f}"
    )

    if return_result:
        return helical


def radDist_SquareTube_parallel(
    args: tuple, return_result: bool = False
):  # -> None | radDist.SquareTube:
    """
    Worker function for parallel computation of Square Tube radial distribution.

    Designed to be called via multiprocessing.Pool.map, which requires a single
    argument. The args tuple is unpacked internally.

    Parameters
    ----------
    args : tuple of (rz_array, config)
        rz_array : array-like of length 2 — (R, z) start coordinates in metres.
        config   : configuration object passed to radDist.SquareTube.
    """
    rzArray, config = args
    squareTube = radDist.SquareTube(startR=rzArray[0], startZ=rzArray[1], config=config)
    squareTube.build()

    logger.info(
        f"DONE with elongatedRing radDist, R = {rzArray[0]:.2f}, z = {rzArray[1]:.2f}"
    )
    print(
        f"DONE with elongatedRing radDist, R = {rzArray[0]:.2f}, z = {rzArray[1]:.2f}"
    )

    if return_result:
        return squareTube


def radDist_NIMROD_parallel(
    args: tuple, return_result: bool = False
):  # -> None | radDist.SquareTube:
    """
    Worker function for parallel computation of NIMROD radial distribution.

    Designed to be called via multiprocessing.Pool.map, which requires a single
    argument. The args tuple is unpacked internally.

    Parameters
    ----------
    args : tuple of (nimrodFile, timestep, config)
        rz_array : array-like of length 2 — (R, z) start coordinates in metres.
        config   : configuration object passed to radDist.SquareTube.
    """
    nimrodFile, timestep, config = args
    nimrod = radDist.NIMROD(nimrodFile=nimrodFile, timestep = timestep, config=config)
    nimrod.build()

    logger.info(
        "DONE with NIMROD radDist"
    )
    print("DONE with NIMROD radDist")
    
    if return_result:
        return nimrod


def callRZGridTokamak(
    tokamak: Tokamak | None = None,
    num_r: int = 30,
    num_z: int = 15,
) -> np.ndarray | None:
    """
    Calls createRZgrid using the tokamak class as an input

    Parameters
    ----------
    tokamak   : Tokamak class instance.
    num_r     : number of equally spaced R points.
    num_z     : number of equally spaced z points.

    Returns
    -------
    rzarray : np.ndarray, shape (N, 2)
        Columns are [R, z]. N ≤ num_r x num_z depending on wall masking.
    """

    rzarray = None
    if tokamak is not None:
        if tokamak.wall is not None:
            rLimits = (tokamak.wall["minr"], tokamak.wall["maxr"])
            zLimits = (tokamak.wall["minz"], tokamak.wall["maxz"])

            rzarray = createRZGrid(
                r_limits=rLimits,
                z_limits=zLimits,
                num_r=num_r,
                num_z=num_z,
                wallcurve=tokamak.wall["wallcurve"],
            )
            return rzarray


def createRZGrid(
    r_limits: tuple[float, float],
    z_limits: tuple[float, float],
    num_r: int = 30,
    num_z: int = 15,
    wallcurve: "matplotlib.path.Path | None" = None,
) -> np.ndarray:
    """
    Create a uniform R-z grid, optionally masked to points inside the wall.

    Parameters
    ----------
    r_limits  : (r_min, r_max) in metres.
    z_limits  : (z_min, z_max) in metres.
    num_r     : number of equally spaced R points.
    num_z     : number of equally spaced z points.
    wallcurve : matplotlib Path defining the wall boundary (from
                Tokamak._load_first_wall()). If None, all grid points
                are returned.

    Returns
    -------
    rzarray : np.ndarray, shape (N, 2)
        Columns are [R, z]. N ≤ num_r x num_z depending on wall masking.
    """

    R_vals = np.linspace(*r_limits, num_r)
    z_vals = np.linspace(*z_limits, num_z)

    RR, ZZ = np.meshgrid(R_vals, z_vals, indexing="ij")
    rzarray = np.column_stack([RR.ravel(), ZZ.ravel()])

    if wallcurve is not None:
        inside = wallcurve.contains_points(rzarray)
        rzarray = rzarray[inside]
    return rzarray


def random_uniform_point_noVolume(Wallcurve, Minr, Maxr, Minz, Maxz):
    """
    Candidate points are drawn uniformly from the bounding box and
    rejected if outside the wall polygon
    """
    x = y = z = r = phi = None  # Initialize variables to ensure they are always defined

    success = 0
    while success == 0:
        x = random.uniform(-Maxr, Maxr)
        y = random.uniform(-Maxr, Maxr)
        z = random.uniform(Minz, Maxz)
        r, phi = Util.XY_To_RPhi(x, y)
        r = np.sqrt((x**2) + (y**2))

        if r < Minr or r > Maxr:
            pass
        elif Wallcurve.contains_points([(r, z)]):
            success = 1

    return x, y, z, r, phi


def bivariate_normal_elongated(
    R: np.ndarray,
    z: np.ndarray,
    R0: float = 0.0,
    z0: float = 0.0,
    sigma_R: float = 1.0,
    sigma_z: float = 1.0,
    theta: float = 0.0,
):
    """
    Rotatable 2-D Gaussian distribution in the poloidal plane.

    Generalises bivariate_normal by allowing an elliptical (elongated)
    and rotated beam profile. Normalised such that ∫∫ f dR dz = 1.

    See: https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function

    Parameters
    ----------
    R, z        : array-like — evaluation coordinates.
    R0, z0      : float      — centre of the distribution.
    sigma_R     : float      — standard deviation along the R axis. Must be > 0.
    sigma_z     : float      — standard deviation along the z axis. Must be > 0.
    theta       : float      — rotation angle in degrees (counter-clockwise from R-axis).

    Returns
    -------
    np.ndarray — density at each (R, z) point.
    """

    # --- Make sure R and z are numpy arrays
    R = np.asarray(R)
    z = np.asarray(z)

    theta_rad = np.deg2rad(theta)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)

    a = cos_t**2 / (2.0 * sigma_R**2) + sin_t**2 / (2.0 * sigma_z**2)

    b = np.sin(2.0 * theta_rad) / (4.0 * sigma_R**2) - np.sin(2.0 * theta_rad) / (
        4.0 * sigma_z**2
    )
    c = sin_t**2 / (2.0 * sigma_R**2) + cos_t**2 / (2.0 * sigma_z**2)

    dR = R - R0
    dz = z - z0

    norm = 1.0 / (2.0 * np.pi * sigma_R * sigma_z)
    exponent = -(a * dR**2 + 2.0 * b * dR * dz + c * dz**2)

    return norm * np.exp(exponent)


def bivariate_normal(
    R: np.ndarray,
    z: np.ndarray,
    R0: float = 0.0,
    z0: float = 0.0,
    sigma_z: float = 1.0,
) -> np.ndarray:
    """
    Bivariate normal distribution in the poloidal plane.

    Normalised such that ∫∫ f dR dz = 1.

    Parameters
    ----------
    R, z      : array-like — evaluation coordinates.
    R0, z0    : float      — centre of the distribution.
    sigma_z : float      — standard deviation (same in R and z). Must be > 0.

    Returns
    -------
    emis : np.ndarray — density at each (R, z) point.
    """

    # --- Ensure that R and z are numpy arrays
    R = np.asarray(R)
    z = np.asarray(z)

    norm = 1 / (2 * np.pi * sigma_z**2)
    exponent = -0.5 * ((R - R0) ** 2 + (z - z0) ** 2) / sigma_z**2

    return norm * np.exp(exponent)


@nb.njit(parallel=True, fastmath=True, cache=True)
def _evaluate_kernels(R, z, R0_arr, z0_arr, weights, sigma_z):
    """
    Fused, parallelised kernel evaluation — avoids allocating any (M, N) matrix.
    Each output point is computed independently in parallel.

    Parameters
    ----------
    R, z       : (N,) float64 — evaluation coordinates.
    R0_arr, z0_arr : (M,) float64 — field line centres.
    weights    : (M,) float64 — per-line weights.
    sigma_z  : float — kernel width.

    Returns
    -------
    tot : (N,) float64
    """
    N = len(R)
    M = len(weights)
    norm = 1.0 / (2.0 * np.pi * sigma_z**2)
    inv_2sig2 = 0.5 / sigma_z**2

    tot = np.zeros(N)
    for n in nb.prange(N):  # parallel over grid points
        val = 0.0
        for m in range(M):  # serial over field lines (stays in cache)
            dR = R[n] - R0_arr[m]
            dz = z[n] - z0_arr[m]
            val += weights[m] * norm * np.exp(-inv_2sig2 * (dR * dR + dz * dz))
        tot[n] = val
    return tot


def bivariate_normal_isodensity_points(
    R0: float,
    z0: float,
    sigma_target: float,
    sigma_kernel: float,
    n: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n source points and weights such that:

        sum_i = w_i * bivariate_normal(R, z, R0=R0_i, z0=z0_i, sigma_z=sigma_kernel)
        ≈ bivariate_normal(R, z, R0=R0, z0=z0, sigma_z=sigma_target)


    Parameters
    ----------
    R0, z0        : float — centre of the target distribution.
    sigma_target  : float — std dev of the target Gaussian (σ_T). Must be > sigma_kernel.
    sigma_kernel  : float — std dev used in each bivariate_normal call (σ_k).
    n             : int   — number of source points.
    seed          : int, optional — RNG seed for reproducibility.

    Returns
    -------
    points  : np.ndarray, shape (n, 2) — source (R, z) coordinates.
    weights : np.ndarray, shape (n,)   — equal weights summing to 1.

    Raises
    ------
    ValueError if sigma_kernel >= sigma_target (convolution can't shrink a Gaussian).
    """
    if sigma_kernel >= sigma_target:
        raise ValueError(
            f"sigma_kernel ({sigma_kernel}) must be strictly less than "
            f"sigma_target ({sigma_target})."
        )

    sigma_source = np.sqrt(sigma_target**2 - sigma_kernel**2)

    rng = np.random.default_rng(seed)
    points = rng.normal(loc=[R0, z0], scale=sigma_source, size=(n, 2))
    weights = np.ones(n) / n

    return points, weights
