# Util.py
"""
File contains general definitions used by many classes
"""

import json
import logging
import os

import re
import h5py
import numpy as np
import yaml
from raysect.core import Point2D
from typing import Any, Union
from pathlib import Path
from raysect.core import Vector3D
from contextlib import contextmanager
import time as _time

logger = logging.getLogger(__name__)


@contextmanager
def phase_timer(label: str, echo: bool = True):
    """
    Coarse wall-clock timer for profiling radDist creation phases.

    Usage:
        with phase_timer("bolos_observe"):
            self.bolos_observe()

    Logs (and by default prints) '[timing] {label}: {dt} s' so the phase
    breakdown is visible even when logging is not configured in scripts.
    """
    t0 = _time.perf_counter()
    try:
        yield
    finally:
        dt = _time.perf_counter() - t0
        msg = f"[timing] {label}: {dt:.2f} s"
        logging.getLogger(__name__).info(msg)
        if echo:
            print(msg, flush=True)


def config_loader(
    pathFileName: Union[str, Path] = "", verbose: bool = False
) -> dict | None:
    try:
        with open(pathFileName, "r") as f:
            config = yaml.safe_load(f)
        if verbose:
            logger.debug("Loaded config file: %s", pathFileName)
        return config

    except Exception as e:
        logger.error("Could not load the configuration file: %s — %s", pathFileName, e)
        return None


def XY_To_RPhi(X: float, Y: float, TorOffset: float = 0.0) -> tuple[float, float]:
    """Convert from the Cartesian x,y coordinates to the major radius and toroidal angle phi.
    Phi is returned in radians and R in meters"""

    R = np.hypot(X, Y)
    phi = np.arctan2(Y, X) - TorOffset

    if phi < -np.pi:
        phi = phi + 2.0 * np.pi
    elif phi > np.pi:
        phi = phi - 2.0 * np.pi

    return R, phi


def RPhi_To_XY(R: float, Phi: float) -> tuple[float, float]:

    x = R * np.cos(Phi)
    y = R * np.sin(Phi)

    return x, y


def rpz_XYZ(rpz: np.ndarray) -> np.ndarray:
    """


    Parameters
    ----------
    rpz : 3xN array of R, phi, and Z coordinates
        This function converts from R, phi, and Z
        to cartesian.

    Returns
    -------
    XYZ.

    """

    if np.shape(rpz)[0] != 3:
        rpz = rpz.T

    if np.shape(rpz)[0] != 3:
        raise ValueError("Input to rpz_XYZ does not start with a dimension of length 3")

    XYZ = rpz * 0.0
    XYZ[0, :] = rpz[0, :] * np.cos(rpz[1, :])
    XYZ[1, :] = rpz[0, :] * np.sin(rpz[1, :])
    XYZ[2, :] = rpz[2, :]

    return XYZ


def xyz_RPZ(XYZ: np.ndarray, PhiOffset: float = 0.0, Sign: float = 1.0) -> np.ndarray:
    """
    Parameters
    ----------
    XYZ : 3xN array of X, Y, and Z
    PhiOffset: Scalar in radians
    Sign: Scalar, should be +1 or -1

    Returns
    -------
    rpz
    """

    if np.shape(XYZ)[0] != 3:
        raise ValueError("Input to xyz_RPZ does not start with a dimenion of length 3")

    rpz = XYZ * 0.0
    rpz[0, :] = np.sqrt(XYZ[0, :] ** 2 + XYZ[1, :] ** 2)
    rpz[1, :] = Sign * np.arctan2(XYZ[1, :], XYZ[0, :]) + PhiOffset
    rpz[2, :] = XYZ[2, :]

    return rpz


def length_along_wall(
    Rwall: np.ndarray, Zwall: np.ndarray, R0: float
) -> tuple[np.ndarray, float]:
    """
    Compute length along the wall
    return Swall, Swall_max
    """
    Nwall = len(Rwall)
    Swall = np.zeros(Nwall)
    dir = 1
    S0 = 0.0  # Ensure S0 is always defined

    Swall[0] = np.sqrt((Rwall[0] - Rwall[-1]) ** 2 + (Zwall[0] - Zwall[-1]) ** 2)
    if Swall[0] > 0:
        S0 = Zwall[-1] / (Zwall[-1] - Zwall[0]) * Swall[0]
        if Zwall[0] < Zwall[-1]:
            dir = 1
            # ccw
        else:
            dir = -1  # cw

    for i in range(1, Nwall):
        Swall[i] = Swall[i - 1] + np.sqrt(
            (Rwall[i] - Rwall[i - 1]) ** 2 + (Zwall[i] - Zwall[i - 1]) ** 2
        )  # length of curve in m
        if (Zwall[i] * Zwall[i - 1] <= 0) & (Rwall[i] < R0):
            t = Zwall[i - 1] / (Zwall[i - 1] - Zwall[i])
            S0 = Swall[i - 1] + t * (Swall[i] - Swall[i - 1])
            if Zwall[i] < Zwall[i - 1]:
                dir = 1  # ccw
            else:
                dir = -1  # cw

    Swall_max = Swall[-1]

    # set direction and Swall = 0 location
    for i in range(Nwall):
        Swall[i] = dir * (Swall[i] - S0)
        if Swall[i] < 0:
            Swall[i] += Swall_max
        if Swall[i] > Swall_max:
            Swall[i] -= Swall_max
        if abs(Swall[i]) < 1e-12:
            Swall[i] = 0

    return Swall, Swall_max


def draw_radial_lines(
    x0: float, y0: float, R: float, angle_deg: float = 0.0, initial_offset: float = 0.0
) -> dict:
    """
    Returns starting and ending points for a line staring at an initial location
    going in a direction angle_deg + initial_offset. Used in create_d3d_observers.py
    """
    answer = {}
    angle_rad = np.radians(angle_deg + initial_offset)

    dx = R * np.cos(angle_rad)
    dy = R * np.sin(angle_rad)

    x_start = x0
    x_end = x0 + dx
    y_start = y0
    y_end = y0 + dy
    answer["x"] = [x_start, x_end]
    answer["y"] = [y_start, y_end]

    return answer


def rZ_to_theta(r: float, z: float, r0: float = 0.0, z0: float = 0.0) -> float | None:
    """
    Returns theta given the initial coordinates.
    """
    theta = None
    dr = r - r0
    dz = z - z0

    if dr == 0:
        return 90.0 if dz > 0 else 270.0

    t_ = np.rad2deg(np.arctan(dz / dr))

    if dr < 0:
        if dz > 0:
            theta = 180.0 - np.abs(t_)
        elif dz <= 0:
            theta = 180.0 + np.abs(t_)
    elif dr > 0:
        if dz < 0:
            theta = 360.0 - np.abs(t_)
        else:
            theta = t_

    return theta


def find_intersection(
    line: dict, vessel_R: np.ndarray, vessel_Z: np.ndarray
) -> tuple[float, float]:
    """
    Used by create_d3d_observers to find where a line, defined by two points
    intersects the vessel

    Returns R, z
    """

    fit = np.poly1d(np.polyfit(line["x"], line["y"], 1))
    x_new = np.linspace(line["x"][0], line["x"][1], 10000)
    y_new = fit(x_new)

    vessel = np.column_stack((vessel_R, vessel_Z))
    fit_new = np.column_stack((x_new, y_new))

    diffs = vessel[:, np.newaxis, :] - fit_new[np.newaxis, :, :]  # shape (N, M, 2)
    dists = np.linalg.norm(diffs, axis=2)
    i, _ = np.unravel_index(np.argmin(dists), dists.shape)

    return vessel[i][0], vessel[i][1]


def unit_vector_from_angle(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    return np.array([np.cos(rad), np.sin(rad)])


def intersect_lines(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray
) -> np.ndarray | None:
    """
    Find intersection of two lines defined by:
    p1 = point on line 1
    d1 = normal to line 1
    p2 = origin
    d2 = dir2

        Line 1: p1 + t * d1
        Line 2: p2 + s * d2
        Returns the intersection point or None if lines are parallel.
    """
    A = np.array([d1, -d2]).T
    b = p2 - p1
    if np.linalg.matrix_rank(A) < 2:
        return None  # Lines are parallel
    t_s = np.linalg.solve(A, b)
    intersection = p1 + t_s[0] * d1
    return intersection


def rz_to_xyz(
    R: np.ndarray, z: np.ndarray, toroidal_angle_rad: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert arrays of (R, z) points to (x, y, z) by rotating around the z-axis
    by a specified toroidal angle (in radians).

    Parameters:
        R (array-like): Radial distances
        z (array-like): z coordinates (same length as R)
        toroidal_angle_rad (float): Angle in radians to rotate around z-axis

    Returns:
        x, y, z: Arrays of Cartesian coordinates
    """
    R = np.asarray(R)
    z = np.asarray(z)

    if R.shape != z.shape:
        raise ValueError("R and z must have the same shape.")

    x = R * np.cos(toroidal_angle_rad)
    y = R * np.sin(toroidal_angle_rad)

    return x, y, z


def rotate_vector(vector: Vector3D, axis: Vector3D, angle_rad: float) -> Vector3D:
    axis = axis.normalise()
    v_parallel = axis * vector.dot(axis)
    v_perp = vector - v_parallel
    v_perp_rot = v_perp * np.cos(angle_rad) + axis.cross(vector) * np.sin(angle_rad)
    return (v_parallel + v_perp_rot).normalise()


def convert_arrays_to_list(obj) -> Any:
    """
    Converts values within a nested dictionary to a list. Typically
    used prior to saving a json file.
    """
    if isinstance(obj, dict):
        return {k: convert_arrays_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_arrays_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_arrays_to_list(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_json(obj: Any, pathFileName: Union[str, Path], saveFileName: str) -> None:
    """
    Saves the obj, creates the directories if needed
    """

    # --- Create the folder if it doesn't exist
    os.makedirs(pathFileName, exist_ok=True)

    # --- Save the file
    savePath = os.path.join(pathFileName, saveFileName)
    with open(savePath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_json(pathFileName: Union[str, Path]) -> dict:
    """
    Loads the json file from the given path
    """

    with open(pathFileName, "r") as f:
        data = json.load(f)
    return data


def get_filenames_in_directory(directory_path: Union[str, Path]) -> list[str]:
    """
    Returns a list of all files within the specified directory and its subdirectories.

    Excludes yaml and .* files.
    """
    files = []
    for root, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename[0] != "." and filename.split(".")[-1] != "yaml":
                files.append(os.path.join(root, filename))
    return files


def read_h5(path: Union[str, Path]) -> dict:
    data = {}
    with h5py.File(path, "r") as f:
        for dset in _traverse_datasets(f):
            cols = dset.strip().split("/")[1:]
            obj = f[dset]
            if isinstance(obj, h5py.Dataset):
                value = obj[()]
            else:
                value = None
            _ensure_path(data=data, path=cols, default=value)
    return data


def _traverse_datasets(hdf_file: h5py.File):
    # Taken from: https://stackoverflow.com/questions/51548551/reading-nested-h5-group-into-numpy-array
    def h5py_dataset_iterator(g, prefix=""):
        for key in g.keys():
            item = g[key]
            path = f"{prefix}/{key}"
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path


def _ensure_path(
    data: dict, path: list[str], default: Any = None, default_func=lambda x: x
) -> dict:
    """
    # Taken from: https://stackoverflow.com/questions/16333296/how-do-you-create-nested-dict-in-python
    Function:

    - Ensures a path exists within a nested dictionary

    Requires:

    - `data`:
        - Type: dict
        - What: A dictionary to check if the path exists
    - `path`:
        - Type: list of strs
        - What: The path to check

    Optional:

    - `default`:
        - Type: any
        - What: The default item to add to a path that does not yet exist
        - Default: None

    - `default_func`:
        - Type: function
        - What: A single input function that takes in the current path item (or default) and adjusts it
        - Default: `lambda x: x` # Returns the value in the dict or the default value if none was present
    """
    if len(path) > 1:
        if path[0] not in data:
            data[path[0]] = {}
        data[path[0]] = _ensure_path(
            data=data[path[0]],
            path=path[1:],
            default=default,
            default_func=default_func,
        )
    else:
        if path[0] not in data:
            data[path[0]] = default
        data[path[0]] = default_func(data[path[0]])
    return data


def find_max_nested_lists(list_: list) -> float:
    """
    Finds the max value within arbitrarily nested lists
    """

    def _flatten(obj):
        for item in obj:
            if isinstance(item, (list, tuple, np.ndarray)):
                yield from _flatten(item)
            else:
                yield item

    return max(_flatten(list_))


def point3d_to_rz(point) -> tuple[float, float]:
    return Point2D(np.hypot(point.x, point.y), point.z)


def point3d_to_phi(point) -> float:
    """
    Toroidal angle of a 3D point, in DEGREES, wrapped to (-180, 180].
    """
    return float(np.rad2deg(np.arctan2(point.y, point.x)))


def wrapped_phi_difference(phi_start: float, phi_end: float) -> float:
    """
    Smallest signed difference between two toroidal angles, in DEGREES,
    wrapped to [-180, 180). An exactly antipodal pair returns -180.

    Wrapping matters because arctan2 returns angles in (-180, 180], so a chord
    crossing that branch cut (e.g. 179.9 -> -179.9) is 0.2 degrees wide, not
    359.8.
    """
    return (float(phi_end) - float(phi_start) + 180.0) % 360.0 - 180.0


def chord_rz_projection(
    r0: float,
    z0: float,
    phi0: float,
    rf: float,
    zf: float,
    phif: float,
    num_points: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project a straight 3D sightline onto the (R, z) plane.

    The chord runs in a straight line through the vessel from (r0, z0, phi0) to
    (rf, zf, phif). When phi0 != phif that line does not lie in a single
    poloidal plane, so its projection R = sqrt(x**2 + y**2) is a curve rather
    than a straight segment. This samples that curve.

    Inputs:
        r0, z0, phi0 :: start of the chord. phi in DEGREES.
        rf, zf, phif :: end of the chord. phi in DEGREES.
        num_points   :: samples along the chord

    Returns:
        (R, z) arrays of length num_points.
    """

    if num_points < 2:
        raise ValueError(f"num_points must be at least 2, got {num_points}")

    t = np.linspace(0.0, 1.0, num_points)

    x0, y0 = RPhi_To_XY(r0, np.deg2rad(phi0))
    xf, yf = RPhi_To_XY(rf, np.deg2rad(phif))

    x = x0 + (xf - x0) * t
    y = y0 + (yf - y0) * t
    z = z0 + (zf - z0) * t

    return np.hypot(x, y), z


def fit_line_to_rz(
    R: np.ndarray, z: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """
    Fit a straight line to a set of (R, z) points and return the segment
    spanning them.

    Uses a total-least-squares (SVD) fit rather than a polynomial one, so a
    vertical chord is handled the same as any other.

    Returns:
        ((R_start, z_start), (R_end, z_end), max_residual)
        where max_residual is the largest perpendicular distance from a point to
        the fitted line, i.e. how much the projected chord bends away from a
        straight segment.
    """

    points = np.column_stack((np.asarray(R, dtype=float), np.asarray(z, dtype=float)))

    if len(points) < 2:
        raise ValueError("Need at least two points to fit a line")

    mean = points.mean(axis=0)
    centered = points - mean

    # --- Principal direction of the point cloud
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]

    # --- Distance of each point along and perpendicular to that direction
    along = centered @ direction
    normal = np.array([-direction[1], direction[0]])
    residuals = np.abs(centered @ normal)

    start = mean + along.min() * direction
    end = mean + along.max() * direction

    return (start[0], start[1]), (end[0], end[1]), float(residuals.max())


def split_revolutions(x, y, z, phi, R, L) -> list:
    """
    Split field line coordinates into revolutions around the torus,
    keeping phi in original coordinates normalized to [0, 2π).
    """

    x, y = map(np.asarray, (x, y))
    z, phi = map(np.asarray, (z, phi))
    R, L = map(np.asarray, (R, L))

    # Shift phi only for indexing revolutions
    phi_shifted = phi - phi[0]

    # Integer revolution index for each point
    rev_index = (phi_shifted // (2 * np.pi)).astype(int)

    revolutions = []
    for i in range(rev_index.min(), rev_index.max() + 1):
        mask = rev_index == i
        if np.any(mask):
            revolutions.append(
                {
                    "x": x[mask],
                    "y": y[mask],
                    "z": z[mask],
                    "R": R[mask],
                    "L": L[mask],
                    "phi": np.mod(phi[mask], 2 * np.pi),
                }
            )

    # Filter out signle-point tail revolutions, occurs when phi[-1] equals
    # N * 2pi, e.g. np.linspace(..., endpoint=True)
    revolutions = [r for r in revolutions if len(r["phi"]) > 1]

    return revolutions


def get_rectangle_corners(rect) -> list:
    """
    Returns 4 world-space corners of a rectangular primitive.
    Assumes rectangle centered at local origin.
    """

    try:
        h = rect.y_width / 2
        w = rect.x_width / 2
    except Exception:
        h = rect.dy / 2
        w = rect.dx / 2

    local_corners = [
        (-w, -h, 0),
        (-w, h, 0),
        (w, -h, 0),
        (w, h, 0),
    ]

    world = rect.to_root()

    corners = []
    for x, y, z in local_corners:
        p = world * rect.centre_point.__class__(x, y, z)
        corners.append(p)

    return corners


def compute_etendue_metric(p1, p2) -> float:
    """
    Simple angular separation metric in RZ plane.
    Larger separation = larger étendue.
    """

    r1, z1 = point3d_to_rz(p1)
    r2, z2 = point3d_to_rz(p2)

    return np.sqrt((r1 - r2) ** 2 + (z1 - z2) ** 2)


def extract_end_numbers(text):
    """Returns the numbers at the end of a string, used
    for plotting routines"""
    # \d+ matches one or more digits
    # $ ensures they are at the very end of the string
    match = re.search(r"\d+$", text)

    return match.group() if match else None


def fieldline_key(phi, degrees: bool = True) -> str:
    """Helper used to find/set the key used for a field line
    starting at the given phi
    """
    value = float(phi)
    if not degrees:
        value = np.rad2deg(value)

    return str(int(round(value)) % 360)
