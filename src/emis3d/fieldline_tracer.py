#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:29:48 2021

@author: ryansweeney

Edited to remove a lot of user specific file value - JLH, Nov. 25th, 2025

Edited to take a gfile input instead of a eqDesk file, since it is already
read within the tokamak class - JLH, July 25, 2025

Updated to use B_phi = F(psi)/R from the g-file (fpol), with the vacuum value
F = fpol[-1] outside the last closed flux surface. The old code used
B_phi = bcentr * rmagx / R, but bcentr is the vacuum field at rcentr, not rmagx,
and the plasma's own contribution to F was ignored. Together with first-order
(Euler) steps, that made the field-line pitch wrong by a few percent, enough
that q = 2 field lines did not close after two transits.

Sign convention: B_phi = -F/R, the same sign the old code used (-bcentr).
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import solve_ivp


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
        mirror_helicity: bool = False,
    ):
        """
        mirror_helicity: flip the toroidal field's sign relative to the
        poloidal field, so field lines wind the opposite way poloidally per
        toroidal turn. A diagnostic of the sign conventions (a g-file not in
        COCOS 1): with the right convention, data from field-aligned
        radiation fit clearly worse with the helicity mirrored.
        """
        self.gfile = gfile
        self.numTor = NumTor
        self.mirror_helicity = bool(mirror_helicity)
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
        zgrid = self.gfile.zmid + np.linspace(-self.gfile.zdim / 2.0, self.gfile.zdim / 2.0, num=self.gfile.ny)
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

        # --- Toroidal field B_phi = -F(psi)/R (sign as in the old code)
        psiI = scipy.interpolate.RectBivariateSpline(Rgrid, zgrid, np.asarray(psi))
        fpol = np.asarray(self.gfile.fpol, dtype=float)
        simagx, sibdry = float(self.gfile.simagx), float(self.gfile.sibdry)
        psin_grid = np.linspace(0.0, 1.0, fpol.size)

        def F_of_psin(psin: float) -> float:
            # Clipping gives fpol[0] at the axis and the vacuum value fpol[-1]
            # outside the last closed flux surface
            return float(np.interp(min(max(psin, 0.0), 1.0), psin_grid, fpol))

        self.psiI = psiI
        self.F_of_psin = F_of_psin

        helicity = 1.0 if self.mirror_helicity else -1.0  # -1: the sign the tracer has always used

        def B_phi(tR: float, tZ: float) -> float:
            psin = float((psiI.ev(tR, tZ) - simagx) / (sibdry - simagx))
            return helicity * F_of_psin(psin) / tR

        self.B_phi = B_phi

        startR = self.startR
        startZ = self.startZ
        startPhi = self.startPhi

        if startR is None or startZ is None or startPhi is None:
            raise ValueError("startR, startZ, and startPhi must all be provided and not None.")
        phi = np.linspace(
            float(startPhi),
            float(startPhi) + 2.0 * float(NumTransits) * np.pi,
            num=int(self.numTor),
        )

        def rhs(tphi, y):
            tR, tZ = y
            bphi = B_phi(tR, tZ)
            return [tR * brI.ev(tR, tZ) / bphi, tR * bzI.ev(tR, tZ) / bphi]

        sol = solve_ivp(
            rhs,
            (phi[0], phi[-1]),
            [float(startR), float(startZ)],
            t_eval=phi,
            method="DOP853",
            rtol=1e-10,
            atol=1e-12,
        )
        if not sol.success or sol.y.shape[1] != phi.size:
            raise RuntimeError(f"Field-line trace from R={startR}, z={startZ} failed: {sol.message}")

        tR, tZ = sol.y
        self.data["R"] = tR
        self.data["z"] = tZ
        self.data["x"] = tR * np.cos(phi)
        self.data["y"] = tR * np.sin(phi)
        dl = np.sqrt(np.diff(self.data["x"]) ** 2 + np.diff(self.data["y"]) ** 2 + np.diff(tZ) ** 2)
        self.data["L"] = np.concatenate([[0.0], np.cumsum(dl)])

        # Normalize to 2pi
        self.data["phi"] = phi
        self.data["phi_wrapped"] = phi % (2.0 * np.pi)

    def plot_Fields(self) -> None:

        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(2, 2, 1)
        c = ax.contourf(self.R, self.Z, self.brI(self.r, self.z), levels=20, vmin=-5, vmax=5.0)
        fig.colorbar(c, ax=ax)
        ax.set_title("$B_R$ (T)")

        ax2 = plt.subplot(2, 2, 2)
        c2 = ax2.contourf(self.R, self.Z, self.bzI(self.r, self.z), levels=20, vmin=-5, vmax=5.0)
        fig.colorbar(c2, ax=ax2)
        ax2.set_title("$B_Z$ (T)")

        ax3 = plt.subplot(2, 2, 3)
        Bphi_grid = np.vectorize(self.B_phi)(self.R, self.Z)
        c3 = ax3.contourf(self.R, self.Z, Bphi_grid, levels=20)
        fig.colorbar(c3, ax=ax3)
        ax3.set_title(r"$B_\phi$ (T)")

        for ax_ in [ax, ax2, ax3]:
            ax_.set_aspect("equal")
            ax_.set_xlabel("R (m)")
            ax_.set_ylabel("Z (m)")

        plt.tight_layout()
        plt.show()
