# Util.py
"""
File contains general definitions used by many classes
"""

import json
import os

import h5py
import numpy as np
import yaml
from raysect.core import Point2D
from typing import Any

# from scipy.stats import chi2, removing to speed up things


def config_loader(pathFileName="", verbose=False):
    try:
        # --- Open the file
        with open(pathFileName, "r") as f:
            config = yaml.safe_load(f)
        if verbose:
            print(f"Loaded config file: {pathFileName}")
        return config

    except Exception as e:
        print(f"Could not load the cofiguration file: {pathFileName}, error: {e}")
        return None


def XY_To_RPhi(X, Y, TorOffset=0.0):
    """Convert from the Cartesian x,y coordinates to the major radius and toroidal angle phi.
    Phi is returned in radians and R in meters"""

    R = np.hypot(X, Y)
    phi = np.arctan2(Y, X) - TorOffset

    if phi < -np.pi:
        phi = phi + 2.0 * np.pi
    elif phi > np.pi:
        phi = phi - 2.0 * np.pi

    return R, phi


def RPhi_To_XY(R, Phi):

    x = R * np.cos(Phi)
    y = R * np.sin(Phi)

    return x, y


"""
def RedChi2_To_Pvalue(RedChi2, Dof):
    return chi2.cdf(x=RedChi2 * Dof, df=Dof)


def Pvalue_To_RedChi2(Pvalue, Dof):
    return chi2.ppf(q=Pvalue, df=Dof) / Dof
"""


def rpz_XYZ(rpz):
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
        raise ValueError("Input to rpz_XYZ does not start with a dimenion of length 3")

    XYZ = rpz * 0.0
    XYZ[0, :] = rpz[0, :] * np.cos(rpz[1, :])
    XYZ[1, :] = rpz[0, :] * np.sin(rpz[1, :])
    XYZ[2, :] = rpz[2, :]

    return XYZ


def xyz_RPZ(XYZ, PhiOffset=0.0, Sign=1.0):
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
    rpz[1, :] = Sign * np.arctan2(XYZ[0, :], XYZ[1, :]) + PhiOffset
    rpz[2, :] = XYZ[2, :]

    return rpz


def length_along_wall(Rwall, Zwall, R0):
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


def draw_radial_lines(x0, y0, R, angle_deg=0.0, initial_offset=0.0):
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


def rZ_to_theta(r, z, r0=None, z0=None):
    """
    Returns theta given the initial coordinates.
    """
    theta = None
    dr = r - r0
    dz = z - z0

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


def find_intersection(line, vessel_R, vessel_Z):
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


def unit_vector_from_angle(deg):
    rad = np.deg2rad(deg)
    return np.array([np.cos(rad), np.sin(rad)])


def intersect_lines(p1, d1, p2, d2):
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


def rz_to_xyz(R, z, toroidal_angle_rad):
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


def rotate_vector(vector, axis, angle_rad):
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


def save_json(obj, pathFileName, saveFileName):
    """
    Saves the obj, creates the directories if needed
    """

    # --- Create the folder if it doesn't exist
    if not os.path.exists(pathFileName):
        os.makedirs(pathFileName)

    # --- Save the file
    savePath = os.path.join(pathFileName, saveFileName)
    with open(savePath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_json(pathFileName):
    """
    Loads the json file from the given path
    """

    with open(pathFileName, "r") as f:
        data = json.load(f)
    return data


def get_filenames_in_directory(directory_path):
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


def read_h5(path):
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


def _traverse_datasets(hdf_file):
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


def _ensure_path(data, path, default=None, default_func=lambda x: x):
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


def find_max_nested_lists(list_):
    """
    Finds the max value within nested lists
    """
    l_ = [item for sublist in list_ for item in sublist]
    return max(l_)


def point3d_to_rz(point):
    return Point2D(np.hypot(point.x, point.y), point.z)


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

    return revolutions
