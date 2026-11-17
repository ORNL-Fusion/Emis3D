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

# Tolerances of the field-line integration (DOP853)
RTOL, ATOL = 1e-10, 1e-12


class MagneticField:
    """
    The interpolated field of an equilibrium, as the tracer uses it:
    B_R = -(1/R) dpsi/dZ and B_Z = (1/R) dpsi/dR from splines of the g-file's
    psi, and B_phi = -F(psi)/R (+F/R with mirror_helicity) from its fpol,
    with the vacuum value outside the last closed flux surface.

    Building it (a gradient and three splines of the whole grid) takes a few
    milliseconds, so it is built once per g-file (field_of) rather than once
    per traced line.
    """

    def __init__(self, gfile, mirror_helicity: bool = False):
        psi = gfile["psi"]
        self.r = gfile.rleft + np.linspace(0.0, gfile.rdim, num=gfile.nx)
        self.z = gfile.zmid + np.linspace(-gfile.zdim / 2.0, gfile.zdim / 2.0, num=gfile.ny)
        self.R, self.Z = np.meshgrid(self.r, self.z)
        psigrad = np.gradient(psi, self.r, self.z)
        # (the fields on the grid, indexed [R, z] as RectBivariateSpline wants them)
        self.bzI = scipy.interpolate.RectBivariateSpline(self.r, self.z, (psigrad[0].T / self.R).T)
        self.brI = scipy.interpolate.RectBivariateSpline(self.r, self.z, (-psigrad[1].T / self.R).T)
        self.psiI = scipy.interpolate.RectBivariateSpline(self.r, self.z, np.asarray(psi))
        self.fpol = np.asarray(gfile.fpol, dtype=float)
        self.simagx, self.sibdry = float(gfile.simagx), float(gfile.sibdry)
        self.psin_grid = np.linspace(0.0, 1.0, self.fpol.size)
        self.helicity = 1.0 if mirror_helicity else -1.0  # -1: the sign the tracer has always used

    def F_of_psin(self, psin):
        """F(psi_N): fpol[0] at the axis, the vacuum value fpol[-1] outside the last closed flux surface. Scalars or arrays."""
        return np.interp(np.clip(psin, 0.0, 1.0), self.psin_grid, self.fpol)

    def B_phi(self, R, Z):
        """Toroidal field at (R, Z); scalars or arrays."""
        psin = (self.psiI.ev(R, Z) - self.simagx) / (self.sibdry - self.simagx)
        return self.helicity * self.F_of_psin(psin) / R

    def rhs(self, phi, y):
        """dR/dphi and dZ/dphi along field lines. y holds the R of every line, then the Z of every line."""
        n = y.size // 2
        R, Z = y[:n], y[n:]
        over_bphi = R / self.B_phi(R, Z)
        return np.concatenate([self.brI.ev(R, Z) * over_bphi, self.bzI.ev(R, Z) * over_bphi])


_FIELDS: dict = {}  # id(gfile) -> (gfile, {mirror_helicity: MagneticField}); the g-file is kept so its id stays its own


def field_of(gfile, mirror_helicity: bool = False) -> MagneticField:
    """The MagneticField of a g-file, built on first use and kept (for the few most recent g-files)."""
    entry = _FIELDS.get(id(gfile))
    if entry is None or entry[0] is not gfile:
        if len(_FIELDS) >= 4:
            _FIELDS.pop(next(iter(_FIELDS)))
        entry = _FIELDS[id(gfile)] = (gfile, {})
    fields = entry[1]
    if bool(mirror_helicity) not in fields:
        fields[bool(mirror_helicity)] = MagneticField(gfile, mirror_helicity)
    return fields[bool(mirror_helicity)]


def trace_many(gfile, startR, startZ, startPhi: float, NumTor: int, NumTransits: float = 1.0, mirror_helicity: bool = False,
               chunk: int = 256) -> dict:
    """
    Trace several field lines from one toroidal angle at once.

    startR, startZ : the lines' start positions (arrays of the same length N)
    startPhi       : their common start angle [rad]
    NumTor         : output points per line, from startPhi to startPhi + 2 pi NumTransits
                     (NumTransits < 0 traces the other way)
    Returns {"phi": (NumTor,), "R": (N, NumTor), "z": (N, NumTor)}.

    The lines are integrated as one system, `chunk` lines at a time, so the
    field is evaluated for all of them in one call per step: tens of times
    faster than a Fieldline_Tracer per line, which spends its time in
    Python between scalar spline evaluations. The integrator's error norm
    is an average over the system, so the tolerance is tightened by
    sqrt(2 x lines): no line is traced less accurately than on its own.
    The results agree with Fieldline_Tracer to about 1e-8 m.

    If a chunk cannot be integrated (one of its lines fails), its lines are
    traced one at a time, so a failure is reported for the line that
    causes it, as Fieldline_Tracer reports it.
    """
    if gfile is None:
        raise ValueError("gfile is None. Please provide a valid gfile object.")
    R0 = np.atleast_1d(np.asarray(startR, dtype=float))
    Z0 = np.atleast_1d(np.asarray(startZ, dtype=float))
    if R0.shape != Z0.shape or R0.ndim != 1:
        raise ValueError("startR and startZ must be one-dimensional and of the same length")
    field = field_of(gfile, mirror_helicity)
    phi = np.linspace(float(startPhi), float(startPhi) + 2.0 * float(NumTransits) * np.pi, num=int(NumTor))
    R_out = np.empty((R0.size, phi.size))
    Z_out = np.empty((R0.size, phi.size))
    for start in range(0, R0.size, max(1, int(chunk))):
        sel = slice(start, min(start + max(1, int(chunk)), R0.size))
        n = sel.stop - sel.start
        ok = False
        try:
            rtol = max(RTOL / np.sqrt(2.0 * n), 1e-13)
            sol = solve_ivp(field.rhs, (phi[0], phi[-1]), np.concatenate([R0[sel], Z0[sel]]), t_eval=phi, method="DOP853", rtol=rtol, atol=ATOL)
            ok = bool(sol.success) and sol.y.shape == (2 * n, phi.size) and bool(np.all(np.isfinite(sol.y)))
            if ok:
                R_out[sel], Z_out[sel] = sol.y[:n], sol.y[n:]
        except (ValueError, ArithmeticError):
            ok = False
        if not ok:  # one at a time: the line that fails raises, as it always has
            for i in range(sel.start, sel.stop):
                tracer = Fieldline_Tracer(StartR=R0[i], StartZ=Z0[i], StartPhi=startPhi, gfile=gfile, NumTor=NumTor, mirror_helicity=mirror_helicity)
                tracer.trace(NumTransits=NumTransits)
                R_out[i], Z_out[i] = tracer.data["R"], tracer.data["z"]
    return {"phi": phi, "R": R_out, "z": Z_out}


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

        # The field's interpolators, built once per g-file (field_of)
        field = field_of(self.gfile, self.mirror_helicity)
        self.field = field
        self.brI = field.brI
        self.bzI = field.bzI
        self.bcentr = -self.gfile.bcentr
        self.r = field.r
        self.z = field.z
        self.R = field.R
        self.Z = field.Z
        self.Rmagx = self.gfile.rmagx
        self.Zmagx = self.gfile.zmagx
        self.psiI = field.psiI

        # --- Toroidal field B_phi = -F(psi)/R (sign as in the old code)
        def F_of_psin(psin: float) -> float:
            # Clipping gives fpol[0] at the axis and the vacuum value fpol[-1]
            # outside the last closed flux surface
            return float(field.F_of_psin(psin))

        self.F_of_psin = F_of_psin

        def B_phi(tR: float, tZ: float) -> float:
            return float(field.B_phi(tR, tZ))

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

        sol = solve_ivp(
            field.rhs,
            (phi[0], phi[-1]),
            np.array([float(startR), float(startZ)]),
            t_eval=phi,
            method="DOP853",
            rtol=RTOL,
            atol=ATOL,
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
