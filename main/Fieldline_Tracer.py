#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:29:48 2021

@author: ryansweeney

Edited to remove a lot of user specific file value - JLH, Nov. 25th, 2025

Edited to take a gfile input instead of a eqDesk file, since it is already
read within the tokamak class - JLH, July 25, 2025

TODO:
1. Why don't the q=2 field lines match up after 2 revolutions? q=1 ones are fine...
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy


class Fieldline_Tracer:
    """
    Class which determines the location of a field line with the given R, z, phi inputs.

    StartR: R start location of the field line
    StartZ: z start location of the field line
    StartPhi: phi starting location of the field line in radians!!
    EqFileName: name of the self.gfile in Emis3D_Inputs/eqdsk, e.g. g183282.01900
    NumTor: Resolution of the field line
    """

    def __init__(
        self,
        StartR=None,
        StartZ=None,
        StartPhi=None,
        gfile=None,
        NumTor=50000,
    ):
        self.gfile = gfile
        self.numTor = NumTor
        self.startR = StartR
        self.startZ = StartZ
        self.startPhi = StartPhi

    def trace(self, NumTransits=1.0) -> None:
        """
        Preforms the field line tracing
        """
        self.data = {}
        if self.gfile is None:
            raise ValueError("gfile is None. Please provide a valid gfile object.")
        psi = self.gfile["psi"]
        Rgrid = self.gfile.rleft + np.linspace(0.0, self.gfile.rdim, num=self.gfile.nx)
        zgrid = self.gfile.zmid + np.linspace(
            -self.gfile.zdim / 2.0, self.gfile.zdim / 2.0, num=self.gfile.ny
        )
        R, Z = np.meshgrid(Rgrid, zgrid)
        psigrad = np.gradient(psi, Rgrid, zgrid)

        """
        # --- interp2d has been deprecated and removed, now using RectBivariateSpline
        # to do this, we need to transpose the original input as well as transpose
        # the answer whenever you call the function, see:

            tBR = brI(tR, tZ).T
            tBZ = bzI(tR, tZ).T

        OLD CODE:
        bzI = scipy.interpolate.interp2d(R, Z, psigrad[0].T / R)
        brI = scipy.interpolate.interp2d(R, Z, -psigrad[1].T / R)
        """

        bzI = scipy.interpolate.RectBivariateSpline(Rgrid, zgrid, (psigrad[0].T / R).T)
        brI = scipy.interpolate.RectBivariateSpline(Rgrid, zgrid, (-psigrad[1].T / R).T)

        self.brI = brI
        self.bzI = bzI
        self.bcentr = -self.gfile.bcentr
        self.r = Rgrid
        self.z = zgrid
        self.R = R
        self.Z = Z
        self.Rmagx = self.gfile.rmagx
        self.Zmagx = self.gfile.zmagx

        # now we need to covert to bx, by, bz. these will be huge arrays, so we should
        # make these functions instead.
        numTor = self.numTor
        for val in ["x", "y", "z", "R", "L"]:
            self.data[val] = np.zeros(numTor)

        startR = self.startR
        startZ = self.startZ
        startPhi = self.startPhi

        if startR is None or startZ is None or startPhi is None:
            raise ValueError(
                "startR, startZ, and startPhi must all be provided and not None."
            )
        phi = np.linspace(
            float(startPhi),
            float(startPhi) + 2.0 * float(NumTransits) * np.pi,
            num=int(numTor),
        )
        dphi = phi[1] - phi[0]

        # initialize this x, this y, this z, and this phi
        tX = float(startR) * np.cos(float(startPhi))
        tY = float(startR) * np.sin(float(startPhi))
        tZ = float(startZ)
        tR = float(startR)

        for i in range(0, numTor):

            tphi = phi[i]

            # update the trace arrays
            self.data["x"][i] = tX
            self.data["y"][i] = tY
            self.data["z"][i] = tZ
            self.data["R"][i] = tR

            tBR = brI(tR, tZ).T
            tBZ = bzI(tR, tZ).T
            tBphi = self.bcentr * self.Rmagx / tR

            dR = tBR / tBphi * dphi * tR
            dZ = tBZ / tBphi * dphi * tR

            dX = -tR * dphi * np.sin(tphi) + dR * np.cos(tphi)
            dY = tR * dphi * np.cos(tphi) + dR * np.sin(tphi)

            dl = np.sqrt(dX**2 + dY**2 + dZ**2)
            if i > 0:
                self.data["L"][i] = self.data["L"][i - 1] + dl

            tX += dX
            tY += dY
            tZ += dZ
            tR += dR

        # Normalize to 2pi
        self.data["phi"] = phi
        self.data["phi_wrapped"] = phi % (2.0 * np.pi)

    def plot_Fields(self) -> None:

        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(2, 2, 1)
        c = ax.contourf(
            self.R, self.Z, self.brI(self.r, self.z), levels=20, vmin=-5, vmax=5.0
        )
        fig.colorbar(c, ax=ax)
        ax.set_title("$B_R$ (T)")

        ax2 = plt.subplot(2, 2, 2)
        c2 = ax2.contourf(
            self.R, self.Z, self.bzI(self.r, self.z), levels=20, vmin=-5, vmax=5.0
        )
        fig.colorbar(c2, ax=ax2)
        ax2.set_title("$B_Z$ (T)")

        ax3 = plt.subplot(2, 2, 3)
        c3 = ax3.contourf(
            self.R, self.Z, self.bcentr * 1.85 / self.R, levels=20, vmin=0, vmax=20.0
        )
        fig.colorbar(c3, ax=ax3)
        ax3.set_title(r"$B_\phi$ (T)")

        for ax_ in [ax, ax2, ax3]:
            ax_.set_aspect("equal")
            ax_.set_xlabel("R (m)")
            ax_.set_ylabel("Z (m)")

        plt.tight_layout()
        plt.show()
