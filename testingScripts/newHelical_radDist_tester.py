# newHelical_radDist_tester.py
"""
Testing out a new newHelical radDistok. This one should expand as it moves away from
the creation point (aka injection location)

Perhaps start out with many field lines, with each of those having a Gaussian distribution?
Then we add them together to get the radDist

This is now the Helical radDist, the old one is now helicalRing
"""


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np

import emis3d.Util_radDist as Util_radDist
from emis3d.Globals import *
from emis3d.workspace import get_workspace
from emis3d.Tokamak import Tokamak
from emis3d.Util import config_loader
import time
import emis3d.radDist as radDist

tokamakName = "DIII-D"
configFileName = "helical_config.yaml"
startPhi = 225
R0_initial = 2.029
z0_initial = 0.409


colors = ["purple", "red", "blue", "green", "purple", "orange"]


def circle_points(R, z, dr, n_points):
    """
    Generate points around (R, z) in a circle of radius dr.

    Parameters:
        R (float): center R coordinate
        z (float): center z coordinate
        dr (float): radius of the circle
        n_points (int): number of points to generate

    Returns:
        list of tuples: [(R1, z1), (R2, z2), ...]
    """
    R_vals = []
    z_vals = []
    for i in range(n_points):
        theta = 2 * np.pi * i / n_points
        R_vals.append(R + dr * np.cos(theta))
        z_vals.append(z + dr * np.sin(theta))

    return R_vals, z_vals


def pellet_initial_parameters(csp=False):
    """
    Gives the basic SPI trajector parameters
    """
    data = {}
    if csp:
        # --- CSP trajectory parameters
        data["R_OUT"] = 2.35  # 2.249
        data["Z_OUT"] = 0.0

        # SPI breaker tube angle down (degrees)
        data["THETA"] = 0
        data["DISP"] = 2

        # SPI length (arbitrary, can increase or decrease)
        data["LENGTH"] = 0.8

    else:
        # --- SPI trajectory parameters
        data["R_OUT"] = 2.284
        # SPI breaker tube tip Z (outer wall)
        data["Z_OUT"] = 0.6845

        # SPI breaker tube angle down (degrees)
        data["THETA"] = 47.3
        data["DISP"] = 15.0

        # SPI length (arbitrary, can increase or decrease)
        data["LENGTH"] = 0.8

    data["Z_IN"] = data["Z_OUT"] - data["LENGTH"] * np.sin(np.deg2rad(data["THETA"]))
    data["R_IN"] = data["R_OUT"] - data["LENGTH"] * np.cos(np.deg2rad(data["THETA"]))

    # --- Find the upper and lower scatter points
    dz = data["LENGTH"] * np.sin((np.deg2rad(data["THETA"] - data["DISP"])))
    dr = data["LENGTH"] * np.cos((np.deg2rad(data["THETA"] - data["DISP"])))
    data["Z_IN_UPPER"] = data["Z_OUT"] - dz
    data["R_IN_UPPER"] = data["R_OUT"] - dr

    dz = data["LENGTH"] * np.sin((np.deg2rad(data["THETA"] + data["DISP"])))
    dr = data["LENGTH"] * np.cos((np.deg2rad(data["THETA"] + data["DISP"])))
    data["Z_IN_LOWER"] = data["Z_OUT"] - dz
    data["R_IN_LOWER"] = data["R_OUT"] - dr

    return data


tok = Tokamak(
    tokamakName=tokamakName,
    mode="Build",
    reflections=False,
    eqFileName="g184407.02100",
    loadBolometers=True,
    verbose=False,
)


sigma_target = 0.06
sigma_kernel = 0.05  # pol_sigma
n = 2_000
points, weights = Util_radDist.bivariate_normal_isodensity_points(
    R0_initial, z0_initial, sigma_target, sigma_kernel, n, seed=42
)
Rvalues = points[:, 0]  # + R0
zValues = points[:, 1]  # + z0


tok.set_fieldlines(
    startR=Rvalues,
    startZ=zValues,
    startPhi=np.deg2rad(startPhi),
    numTransists=1.0,
)

# --- Initial test at the injection location
if False:
    emissionName = "counterClock_rev0"
    RZarray = Util_radDist.callRZGridTokamak(tok, numRgrid=500, numZgrid=500)

    # --- Test the current implementation of a radDist
    # --- Create the radDist using only one point, we don't need to loop over everything
    pathFileName = os.path.join(
        get_workspace().path("raddists", tokamak=tokamakName), configFileName
    )
    config = config_loader(pathFileName)
    if config is None:
        raise FileNotFoundError(f"Could not load config file: {pathFileName}")
    config["sigma_z"] = sigma_kernel
    config["numFieldLines"] = n
    rD = radDist.Helical(startR=R0_initial, startZ=z0_initial, config=config)

    R = RZarray[:, 0]
    z = RZarray[:, 1]
    phi = np.array([np.deg2rad(startPhi)])

    ans = rD.calc_emissivity(R, z, phi, emissionName=emissionName)
    tot_rD = ans[emissionName]

    # --- Each field line will be a bivariate normal, let's caclculate the
    # scale factor for points that are not in the center
    R0_arr, z0_arr = rD.tokamak.find_RZ_Fline(
        str(startPhi),
        emissionName=emissionName,
        inputPhis=[np.deg2rad(startPhi)],
    )

    R = np.ascontiguousarray(R.ravel(), dtype=np.float64)
    z = np.ascontiguousarray(z.ravel(), dtype=np.float64)
    R0_arr = np.ascontiguousarray(R0_arr.ravel(), dtype=np.float64)
    z0_arr = np.ascontiguousarray(z0_arr.ravel(), dtype=np.float64)

    tot2 = Util_radDist._evaluate_kernels(
        RZarray[:, 0], RZarray[:, 1], R, z, rD.info["weights"], pol_sigma=sigma_kernel
    )

    # --- This function will be replaced by the _evalulate call, using x, y, and z cherab inputs
    # instead of a pre-defined grid
    tot = np.zeros(len(RZarray))
    for (Ri, zi), w, R0, z0 in zip(points, weights, R0_arr, z0_arr):
        tot += w * Util_radDist.bivariate_normal(
            R=RZarray[:, 0], z=RZarray[:, 1], R0=R0, z0=z0, pol_sigma=sigma_kernel
        )

    plt.ion()
    f = plt.figure(figsize=(8, 6))
    ax = f.add_subplot(121)
    ax1 = f.add_subplot(122)
    tok._plot_first_wall(ax)
    tok._plot_first_wall(ax1)
    R_unique = RZarray[:, 0]
    z_unique = RZarray[:, 1]
    n_levels = 50

    cf = ax.tricontourf(R_unique, z_unique, tot, levels=n_levels, cmap="plasma")
    ax.tricontour(
        R_unique,
        z_unique,
        tot,
        levels=n_levels,
        colors="white",
        linewidths=0.4,
        alpha=0.4,
    )
    ax.set_title("Manual")
    plt.colorbar(cf, ax=ax, label="Density")

    cf = ax1.tricontourf(R_unique, z_unique, tot_rD, levels=n_levels, cmap="plasma")
    ax1.tricontour(
        R_unique,
        z_unique,
        tot_rD,
        levels=n_levels,
        colors="white",
        linewidths=0.4,
        alpha=0.4,
    )
    ax1.set_title("From radDist class")
    plt.colorbar(cf, ax=ax1, label="Density")
    plt.tight_layout()
    plt.show()
    print(f"Difference: {np.sum(tot_rD - tot)}")

# --- Try it out at the bolometer locations, compare new old vectorization techniques
if False:
    # --- Plot the field lines at the relevant locations
    phis = [startPhi, 270, 270, 135]
    groupNames = ["", "SX90PF", "SX90MF", "DISRADU"]
    plt.ion()
    f, ax = plt.subplots(figsize=(18, 8), ncols=4, nrows=1)

    emissionNames = tok.fieldLines[str(startPhi)]["directionNames"]

    for ii, ax_ in enumerate(ax.flat):

        # --- Find the simulated radiation at that location
        RZarray = Util_radDist.callRZGridTokamak(tok, numRgrid=500, numZgrid=500)

        tot = np.zeros(len(RZarray))
        tot_vec = np.zeros(len(RZarray))
        for qq, emissionName in enumerate(emissionNames):
            R0_arr, z0_arr = tok.find_RZ_Fline(
                str(startPhi),
                emissionName=emissionName,
                inputPhis=[np.deg2rad(phis[ii])],
            )

            weights = weights
            pol_sigma = sigma_kernel

            R = RZarray[:, 0]
            z = RZarray[:, 1]

            # --- Trying to speed everything up
            t0 = time.time()

            # Squeeze out any spurious dimensions, ensure C-contiguous float64
            R = np.ascontiguousarray(R.ravel(), dtype=np.float64)
            z = np.ascontiguousarray(z.ravel(), dtype=np.float64)
            R0_arr = np.ascontiguousarray(R0_arr.ravel(), dtype=np.float64)
            z0_arr = np.ascontiguousarray(z0_arr.ravel(), dtype=np.float64)

            tot_vec += Util_radDist._evaluate_kernels(
                R,
                z,
                R0_arr,
                z0_arr,
                weights,
                pol_sigma,
            )

            tf = time.time() - t0

            # --- Old method -- used a for loop
            t02 = time.time()
            for w, R0, z0 in zip(weights, R0_arr, z0_arr):
                tot += w * Util_radDist.bivariate_normal(
                    R=RZarray[:, 0],
                    z=RZarray[:, 1],
                    R0=R0,
                    z0=z0,
                    pol_sigma=sigma_kernel,
                )
            tf2 = time.time() - t02

            # --- Sanity check both methods agree
            print(f"New: {tf:.3f} s, Old: {tf2:.3f} s")
            print(f"Max diff between methods: {np.abs(tot_vec - tot).max():.2e}")

            R_unique = RZarray[:, 0]
            z_unique = RZarray[:, 1]
            n_levels = 50

            cf = ax_.tricontourf(
                R_unique, z_unique, tot_vec, levels=n_levels, cmap="CMRmap_r"
            )
            ax_.tricontour(
                R_unique,
                z_unique,
                tot_vec,
                levels=n_levels,
                colors="white",
                linewidths=0.4,
                alpha=0.4,
            )

            tok._plot_first_wall(ax_)

            # --- Injection location
            if ii == 0:
                spi_path = pellet_initial_parameters(csp=False)
                ax_.plot(
                    [spi_path["R_IN"], spi_path["R_OUT"]],
                    [spi_path["Z_IN"], spi_path["Z_OUT"]],
                    "-r",
                )
                ax_.plot(
                    [spi_path["R_IN_UPPER"], spi_path["R_OUT"]],
                    [spi_path["Z_IN_UPPER"], spi_path["Z_OUT"]],
                    "-r",
                )
                ax_.plot(
                    [spi_path["R_IN_LOWER"], spi_path["R_OUT"]],
                    [spi_path["Z_IN_LOWER"], spi_path["Z_OUT"]],
                    "-r",
                )

                ax_.plot(tok.gfile.rbbbs, tok.gfile.zbbbs, color="teal")
                ax_.set_title("Injection location")

            # --- Plot the initial points
            if False:
                for jj in range(0, len(R.flatten())):
                    label = "__no_legend__"
                    if jj == 0:
                        label = emissionName

                    ax_.scatter(
                        R[jj],
                        z[jj],
                        color=colors[qq],
                        marker="s",
                        label=label,
                        zorder=10,
                    )

            # --- Plot the bolometers
            if ii > 0:
                tok._plot_bolometers(ax_, boloGroupName=groupNames[ii])

    ax[1].legend()
    plt.tight_layout()
    plt.show()
