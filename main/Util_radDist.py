# Util_RadDist.py
"""
File contains defintions used while creating radDists

Written by JLH Aug. 2025
"""


import random

import numpy as np

import main.radDist as radDist
import main.Util as Util


def radDist_ElongatedRing_parallel(input) -> None:
    """
    Computes Elongated Ring radDists based on input rzArray
    """
    rzArray = input[0]
    config = input[1]
    elongatedRing = radDist.ElongatedRing(
        startR=rzArray[0], startZ=rzArray[1], config=config
    )
    elongatedRing.build()

    print(
        f"DONE with elongatedRing radDist, R = {rzArray[0]:.2f}m, z = {rzArray[1]:.2f}m"
    )


def radDist_Helical_parallel(input) -> None:
    """
    Computes helical radDists based on input rzArray
    """

    rzArray = input[0]
    config = input[1]
    helical = radDist.Helical(startR=rzArray[0], startZ=rzArray[1], config=config)
    helical.build()

    print(f"DONE with helical radDist, R = {rzArray[0]:.2f}m, z = {rzArray[1]:.2f}m")


def radDist_Helical_parallel_return_radDist(input):
    """
    Computes helical radDists based on input rzArray
    """

    rzArray = input[0]
    config = input[1]
    helical = radDist.Helical(startR=rzArray[0], startZ=rzArray[1], config=config)
    helical.build()

    print(f"DONE with helical radDist, R = {rzArray[0]:.2f}m, z = {rzArray[1]:.2f}m")
    return helical


def callRZGridTokamak(tokamak, numRgrid=30, numZgrid=15) -> np.ndarray:
    """
    Calls createRZgrid using the tokamak class as an input
    """
    rLimits = [tokamak.wall["minr"], tokamak.wall["maxr"]]
    zLimits = [tokamak.wall["minz"], tokamak.wall["maxz"]]

    rzarray = createRZGrid(
        rLimits=rLimits,
        zLimits=zLimits,
        numRgrid=numRgrid,
        numZgrid=numZgrid,
        wallcurve=tokamak.wall["wallcurve"],
    )
    return rzarray


def createRZGrid(
    rLimits=[0, 1], zLimits=[-1, 1], numRgrid=30, numZgrid=15, wallcurve=None
) -> np.ndarray:
    """
    Creates a equally spaced R, z grid within the given limits.
    The program will eliminate points outside of the wall, if
    the wallcurve is specified (from the Tokamak class)

    INPUTS

    rLimits :: List of R min and R max points
    zLimits :: List of z min and z max points
    wallcurve :: Created within the tokamak class when _load_first_wall() is called.
                 Type: path.Path(rzarray)
    numRgrid :: The number of equally spaced points within the R grid
    numZgrid :: The number of equally spaced points within the z grid
    """

    rarray = np.linspace(rLimits[0], rLimits[1], num=numRgrid)
    zarray = np.linspace(zLimits[0], zLimits[1], num=numZgrid)

    # we now have our RZ grid specified. We will now reduce the size by
    # omitting points outside the first wall
    rzarray = []
    for j in range(numZgrid):
        for i in range(numRgrid):
            if wallcurve is not None:
                # --- Only save the point if it is inside of the wall
                if wallcurve.contains_points([(rarray[i], zarray[j])]):
                    rzarray.append([rarray[i], zarray[j]])
            else:
                rzarray.append([rarray[i], zarray[j]])

    return np.array(rzarray)


def random_uniform_point_noVolume(Wallcurve, Minr, Maxr, Minz, Maxz):

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


def bivariate_normal_elongated2(
    R=0, R0=0, z=0, z0=0, elongation=0, polSigma=0, theta=0
):
    """
    Bivariate normal distribution in poloidal plane.
    """
    emis = (
        (1.0 / (2.0 * np.pi * elongation * polSigma**2))
        * np.exp(-0.5 * ((R - R0) ** 2) / polSigma**2)
        * np.exp(-0.5 * ((z - z0) ** 2) / (polSigma * elongation) ** 2)
    )

    if theta != 0:
        thetaRad = np.deg2rad(theta)
        c, s = np.cos(thetaRad), np.sin(thetaRad)
        rotation_matrix = np.array([[c, -s], [s, c]])
        temp_ = np.dot(rotation_matrix, emis)
        emis = temp_

    return emis


def bivariate_normal_elongated(R=0, R0=0, z=0, z0=0, elongation=0, polSigma=0, theta=0):
    """
    Guassian distribution that is rotatable, from:
    https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    """
    theta = np.deg2rad(theta)

    a = np.cos(theta) ** 2 / (2.0 * elongation**2)
    b = np.sin(2 * theta) / (4 * elongation**2) - np.sin(2 * theta) / (4 * polSigma**2)
    c = np.sin(theta) ** 2 / (2 * elongation**2) + np.cos(theta) ** 2 / (
        2 * polSigma**2
    )

    emis = (
        1.0
        / (2.0 * np.pi * elongation * polSigma)
        * np.exp(
            -(a * (R - R0) ** 2 + 2.0 * b * (R - R0) * (z - z0) + c * (z - z0) ** 2)
        )
    )

    return emis


def bivariate_normal(R=0, R0=0, z=0, z0=0, polSigma=0):
    """
    Bivariate normal distribution in poloidal plane.
    Integrated over dR and dZ this function returns 1.
    """
    emis = (
        1
        / (2 * np.pi * polSigma**2)
        * np.exp(-0.5 * ((R - R0) ** 2 + (z - z0) ** 2) / polSigma**2)
    )
    return emis
