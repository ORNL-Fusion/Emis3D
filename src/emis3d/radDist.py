# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:12:06 2021

@author: bemst

Re-organized pretty much everything, added parallelization, and a lot more during the refactor -JLH Aug., 2025

Added new Helical class that expands with the field lines, moved old helical class to helicalRing -JLH Mar. 2026

TODO: Update powerPerBin and uncomment it from the self.build()

"""

import logging
import os

import numpy as np
from cherab.tools.emitters import RadiationFunction
from cherab.core.math import AxisymmetricMapper, CylindricalTransform, Interpolate2DLinear
from raysect.optical import VolumeTransform  # type: ignore
from raysect.core.workflow import SerialEngine
from raysect.core.math.random import seed as _raysect_seed

import emis3d.util_radDist as Util_radDist

logger = logging.getLogger(__name__)
from emis3d.globals import RADDIST_FORMAT
from emis3d.workspace import get_workspace
from emis3d.tokamak import Tokamak
from emis3d.util import (
    XY_To_RPhi,
    convert_arrays_to_list,
    save_json,
    load_json,
    # phase_timer,
    fieldline_key,
)
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from abc import ABC, abstractmethod
from pathlib import Path

# For NIMROD class
import json
from typing import Any, cast
from cherab.core.math.interpolators.interpolators3d import Interpolate3DLinear


def _name_value(value) -> str:
    """A parameter value as it appears in a file or folder name."""
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return f"{float(value):g}"
    return str(value).replace("/", "-").replace(" ", "")


class RadDist(ABC):
    """Parent RadDist class."""

    # --- (info key, label) pairs that make up the save folder name
    SAVE_KEYS: tuple[tuple[str, str], ...] = ()
    # Emission that does not depend on the toroidal angle (the fit applies the
    # toroidal shape afterwards): one compiled grid serves every camera, and
    # the toroidal power profile is computed once
    TOROIDALLY_UNIFORM: bool = False
    # A camera whose central sightlines span less than this toroidally is
    # treated as poloidal: its view is one toroidal plane
    POLOIDAL_SPAN_DEG: float = 0.5

    def __init__(self, startR=2.96, startZ=0.0, config=None):

        self.info = dict(config) if config else {}
        self.info["startR"] = startR
        self.info["startZ"] = startZ
        # Ray-integration step [m] (a config value wins; Cherab's default is 0.1 m)
        self.info.setdefault("integrationStep", 0.01)

        if "injectionLocation" in self.info:
            self.info["startPhi"] = self.info["injectionLocation"]
            self.info["startPhiRad"] = np.deg2rad(self.info["injectionLocation"])

        # Holds computed results; initialized here so it always exists.
        self.data = {}

    def _build_tokamak(
        self,
        tokamakName="",
        mode="Build",
        reflections=False,
        eqFileName=None,
        loadBolometers=True,
    ) -> None:
        """
        Initializes an instance of the tokamak class
        """
        # TIMER ADDITION
        # with phase_timer("tokamak build (CAD load + scene graph)"):
        self.tokamak = Tokamak(
            tokamakName=tokamakName,
            mode=mode,
            reflections=reflections,
            eqFileName=eqFileName,
            loadBolometers=loadBolometers,
        )

    def _evaluate_cherab(self, X, Y, Z) -> np.ndarray:
        """
        Wrapper function for self.calc_emissivity to take inputs from Cherab
        """

        R, phi = XY_To_RPhi(X, Y)
        # --- Convert phi to be positive
        if phi is not None and phi < 0:
            phi += 2.0 * np.pi
        emission = self.calc_emissivity(
            np.array([R]),
            np.array([Z]),
            np.array([phi]),
            emissionName=self.emissionName,
        )
        return emission[self.emissionName].item()

    def _update_bolometer_properties(self) -> None:
        """
        Changes Cherab observation parameters. These should be defined within the radDist
        config file under BOLOMETER_PROPS

        Required values in config file:
        pixelSamples    :: Resolution of the sightline. Higher = better, but takes longer.

        Optional values:
        seed            :: Integer RNG seed. When set, observations render in-process
                           so the result is reproducible (a single deterministic draw).

        """

        boloCameras = self.tokamak.bolometers
        # TODO: pixelSamples, numProcessors, and seed are raysect/observe options,
        # not bolometer properties. Move them into a dedicated config group (e.g.
        # observe_opts) instead of overloading BOLOMETER_PROPS.
        pixelSamples = self.info["BOLOMETER_PROPS"]["pixelSamples"]

        seed = self.info["BOLOMETER_PROPS"].get("seed")
        if seed is not None:
            # Seed once before observing.
            #
            # Only when used in combination with a serial rendering engine does this
            # lead to reproducible observes.
            #
            # See: https://github.com/ORNL-Fusion/Emis3D/issues/29
            _raysect_seed(seed)

        for bolo_ in boloCameras:
            # --- Either does the top or bottom loop depending on on if there is an extra bolometerCamera layer
            if hasattr(bolo_, "bolometer_camera"):
                foils = list(bolo_.bolometer_camera.foil_detectors)
            elif hasattr(bolo_, "foil_detectors"):
                foils = list(bolo_.foil_detectors)
            else:
                logger.error("Could not update bolometer properties in RadDist._update_bolometer_properties()")
                foils = []
            for foil in foils:
                # SerialEngine required since user typically uses make_radDists(), which already uses multiple processors
                foil.render_engine = SerialEngine()

                foil.pixel_samples = pixelSamples

    def _get_observed_phi_loc(self) -> None:
        """
        Creates a dict containing the toroidal angle(s) at which each synthetic
        signal was observed, to be used in the fitting. Each channel holds a
        list of angles, one per toroidal segment of the observation (a single
        entry for the standard poloidal-fan case; toroidally-viewing chords will
        populate multiple segments).

        Form:
        {
            emissionName1: {
                bolo1: [[ch1_phi1, ch1_phi2, ...], [ch2_phi1, ...], ...],
                bolo2: [...],
            },
            emissionName2: {...},
             ...
        }
        """

        if getattr(self, "_phi_loc_from_segments", False):
            return  # observe_with_matrix stored each segment's own angle
        boloCameras = self.tokamak.bolometers
        observed_phi_loc = {}
        for emissionName in self.info["emissionNames"]:
            temp = {}
            for bolo_ in boloCameras:
                temp[bolo_.name] = self._observed_phi_loc(bolo_.info, emissionName=emissionName)
            observed_phi_loc[emissionName] = temp
        self.data["observed_phi_loc"] = observed_phi_loc

    @abstractmethod
    def _evaluate(
        self,
        R: np.ndarray,
        z: np.ndarray,
        phi: np.ndarray,
        emissionName: str | None = None,
    ) -> dict:
        """
        Abstract method to be implemented by subclasses to return the emissivity at the given R, z, and phi.
        """
        pass

    def calc_emissivity(
        self,
        R: np.ndarray,
        z: np.ndarray,
        phi: np.ndarray,
        emissionName: str | None = None,
    ) -> dict:
        """
        Broadcast R, z, phi to a common length, then return emissivity from _evaluate.

        Any points outside the tokamak wall are masked to zero.

        Parameters
        ----------
        R, z, phi    : array-like — evaluation coordinates. Each may be length 1
                    (broadcast to the length of the others) or all the same length.
        emissionName : str, optional — defaults to self.emissionName if not given.

        Returns
        -------
        dict of {emissionName: np.ndarray}
        """

        # --- First call _evaluate to get the emissivity, then check if it is inside the tokamak, if not return 0
        # emiss format: {emissionName: np.array[self._evalutate(R0,z0,phi0) self._evalutate(R1,z1,phi1, ...]}
        # aka, it is an array of emission at each R, z, and phi location
        emiss = self._evaluate(R, z, phi, emissionName=emissionName)

        # --- Zero out points outside the tokamak wall
        inside = self.tokamak._inside_tokamak(np.column_stack([R, z]))
        for name in emiss:
            emiss[name][~inside] = 0.0

        return emiss

    def build(self) -> None:
        """
        Creates radDist, finds the power per bin, observes the world for
        each bolometer, then saves the data.

        The tokamak should be created by the individual radDist subtypes
        (helical, elongated ring, etc.) during startup.
        """

        self.calc_radiated_power()
        if str(self.info.get("rayMethod", "compiled")).lower() == "matrix":
            from emis3d.geometry import cached_matrix

            self.observe_with_matrix(cached_matrix(self.info["matrixPath"]))
        else:
            self.bolos_observe()
        self._get_observed_phi_loc()
        # with phase_timer("build/saveRadDist"):
        self.saveRadDist()

    def _total_radiated_power(
        self,
        n_phi: int = 100,
        n_poloidal: int = 200,
        emissionName: str | None = None,
    ) -> dict:
        """
        Compute total radiated power by deterministic quadrature.

        Evaluates the poloidal integral P_pol(φ) at each toroidal location:
            P_pol(φ) = ∫∫_wall ε(R, z, φ) · R dR dz

        Parameters
        ----------
        n_phi      : int — number of toroidal quadrature points.
        n_poloidal : int — number of points along each poloidal axis.
        emissionName : str, optional — defaults to self.emissionName.

        Returns
        -------
        dict with keys:
            phi_array : (n_phi,)  toroidal angles evaluated (radians).
            P_pol     : (n_phi,)  poloidal integral at each phi.
            P_total   : float     total radiated power assuming toroidal symmetry (W).
        """

        wall = self.tokamak.wall

        if wall is None:
            raise RuntimeError("Initilize the tokamak prior to calling _total_radiated_power")

        # -- Full grid creation
        R_vals = np.linspace(wall["minr"], wall["maxr"], n_poloidal)
        z_vals = np.linspace(wall["minz"], wall["maxz"], n_poloidal)
        RR, ZZ = np.meshgrid(R_vals, z_vals, indexing="ij")
        R_flat = RR.ravel()
        z_flat = ZZ.ravel()

        phi_array = np.linspace(0, 2.0 * np.pi, n_phi, endpoint=False)

        # --- Poloidal integral P_pol(φ) at each toroidal location (once, if the
        # emission does not depend on the toroidal angle)
        P_pol = np.zeros(n_phi)

        for i, phi in enumerate(phi_array[:1] if self.TOROIDALLY_UNIFORM else phi_array):

            phi_arr = np.full(len(R_flat), phi)

            # --- Caclulate the emissivity at that toroidal location
            result = self.calc_emissivity(R_flat, z_flat, phi_arr, emissionName=emissionName)
            emiss = result[emissionName]  # (N_inside,)

            # Reshape grid for integration
            emis_grid = (emiss * R_flat).reshape(n_poloidal, n_poloidal)

            inner = simpson(emis_grid, x=z_vals, axis=1)  # (n_poloidal,)
            P_pol[i] = simpson(inner, x=R_vals)

        if self.TOROIDALLY_UNIFORM:
            P_pol[:] = P_pol[0]

        # --- Total power: phi_array is a uniform periodic grid, so the integral
        # over the full turn is the mean times 2 pi (a Simpson rule over
        # phi_array would miss the interval from the last sample back to 2 pi)
        P_total = float(np.mean(P_pol) * 2.0 * np.pi)

        return {
            "phi_array": phi_array,
            "P_pol": P_pol,
            "P_total": P_total,
        }

    def calc_radiated_power(self):
        """
        Calculates the radiated power around the vessel

        """

        self.data = {}
        self.data["toroidalRadiatedPower"] = {}
        for emissionName in self.info["emissionNames"]:
            self.data["toroidalRadiatedPower"][emissionName] = self._total_radiated_power(emissionName=emissionName, n_phi=int(self.info.get("numBins", 100)))

    def power_per_bin_calc(self, Errfrac: float = 0.01, Pointsupdate: int = int(1e5)) -> None:
        """
        Replaces the power_per_bin_calc with a vectorized version.
        """

        # --- Initialize items
        self.data = {}
        self.data["emisSumArray"] = {}
        self.data["emisSqArray"] = {}
        self.data["powerPerBin"] = {}
        self.data["phi"] = []

        numBins = self.info["numBins"]
        angleperbin = 2.0 * np.pi / numBins

        # Ensure tokamak and its info are initialized
        if not hasattr(self, "tokamak") or self.tokamak is None:
            raise RuntimeError("Tokamak object is not initialized. Call _build_tokamak() before power_per_bin_calc().")
        if not hasattr(self.tokamak, "info") or self.tokamak.info is None:
            raise RuntimeError("Tokamak.info is not initialized. Ensure _build_tokamak() sets up info correctly.")

        volumeperbin = self.tokamak.info["MACHINE"]["volume"] / float(numBins)
        pointsperbin = 0
        reachedprecision = 0

        while reachedprecision == 0:
            pointsperbin += Pointsupdate
            # --- Create all of the random points first
            x_, y_, z_, R_, phifirstbin_ = [], [], [], [], []
            while len(x_) < Pointsupdate:
                if self.tokamak.wall is None:
                    raise RuntimeError("Tokamak wall is not initialized. Ensure that the wall attribute is set before calling power_per_bin_calc().")
                # Find random point within the wallcurve
                x, y, z, R, phi = Util_radDist.random_uniform_point_noVolume(
                    self.tokamak.wall["wallcurve"],
                    self.tokamak.wall["minr"],
                    self.tokamak.wall["maxr"],
                    self.tokamak.wall["minz"],
                    self.tokamak.wall["maxz"],
                )
                x_.append(x)
                y_.append(y)
                z_.append(z)
                R_.append(R)

                # --- Program rotates all points to be within the first bin,
                # finds the emissivity, then rotates the same points to each
                # subsequent bin
                if phi is not None and phi < 0:
                    phi += 2.0 * np.pi
                phibin = np.floor(phi / angleperbin)
                phifirstbin_.append(phi - (angleperbin * phibin))

            # --- Evalulate all of the points at once
            R_ = np.array(R_)
            z_ = np.array(z_)

            for numbin in range(0, numBins):
                # print(f"Calculating powerPerBin {numbin + 1} of {numBins}")

                # --- use the initial point for each toroidal bin
                phi = np.array(phifirstbin_) + (angleperbin * numbin)
                self.data["phi"].append(phi)

                # Add the emission to the existing arrays
                for emissionName in self.info["emissionNames"]:

                    emission = self.calc_emissivity(R_, z_, phi, emissionName=emissionName)

                    if emissionName not in self.data["emisSqArray"]:
                        self.data["emisSqArray"][emissionName] = np.zeros(numBins)
                        self.data["emisSumArray"][emissionName] = np.zeros(numBins)
                        self.data["powerPerBin"][emissionName] = np.zeros(numBins)

                    self.data["emisSumArray"][emissionName][numbin] += emission[emissionName].sum()
                    self.data["emisSqArray"][emissionName][numbin] += (emission[emissionName] ** 2).sum()

            # --- Check to see if the desired precision is reached
            # I believe they are checking to see if the variance between bins is small? -JLH
            emismeanarray = {}
            emisvararray = {}
            integemisarray = {}
            totintegemis = 0
            integemisvararray = {}
            integemiserrarray = {}
            totintegemiserr = 0

            for key_ in self.data["emisSumArray"]:
                emismeanarray[key_] = self.data["emisSumArray"][key_] / pointsperbin
                emisvararray[key_] = (self.data["emisSqArray"][key_] / pointsperbin) - (emismeanarray[key_] ** 2)
                integemisarray[key_] = volumeperbin * emismeanarray[key_]

                totintegemis += np.sum(integemisarray[key_])

                integemisvararray[key_] = volumeperbin**2 * emisvararray[key_] / pointsperbin
                integemiserrarray[key_] = np.sqrt(integemisvararray[key_])
                totintegemiserr += np.sum(integemiserrarray[key_])
            toterrfrac = totintegemiserr / totintegemis

            if toterrfrac < Errfrac:
                reachedprecision = 1
                for key_ in integemisarray:
                    self.data["powerPerBin"][key_] = integemisarray[key_]
            else:
                if pointsperbin % (Pointsupdate * 100) == 0:
                    logger.debug(
                        "Number of points %d, total std. err fraction so far = %.2e",
                        pointsperbin,
                        toterrfrac,
                    )

    # ------------------------------------------------------------------
    # Compiled emission (rays: {method: compiled})
    # ------------------------------------------------------------------
    def _grid_spacing(self) -> float:
        """
        Grid spacing [m] for compiled emission: an eighth of the narrowest width
        that shapes the emission, between 0.5 and 3 mm; gridSpacing in the
        settings overrides it.
        """
        if self.info.get("gridSpacing"):
            return float(self.info["gridSpacing"])
        widths = [float(self.info[k]) for k in ("sigma_R", "sigma_z", "sigmaKernel", "dR", "dz")
                  if isinstance(self.info.get(k), (int, float)) and self.info[k] > 0]
        if self.info.get("distType", "").lower() == "helicalring" and self.info.get("sigma_R") and self.info.get("sigma_z"):
            widths.append(float(self.info["sigma_R"]) * float(self.info["sigma_z"]))  # its perpendicular width
        return float(np.clip(min(widths) / 8.0, 5e-4, 3e-3)) if widths else 3e-3

    def _emission_box(self, emissionName: str, phi: float) -> tuple[float, float, float, float] | None:
        """(R_min, R_max, z_min, z_max) outside which this emission is negligible at phi; None: the whole wall."""
        return None

    def _compiled_emission(self, emissionName: str, phi: float = 0.0):
        """
        The emission as a compiled Cherab function, so ray marching does not
        call Python at every step: the emissivity on an (R, z) grid at the
        toroidal angle phi, interpolated by Cherab and mapped around the torus
        (exact for TOROIDALLY_UNIFORM types; for others, valid for a camera
        whose view lies in the plane at phi). The grid covers _emission_box,
        with a zero border, so the emission is zero outside it.
        """
        wall = self.tokamak.wall
        if wall is None:
            raise RuntimeError("Build the tokamak before compiling the emission")
        h = self._grid_spacing()
        box = self._emission_box(emissionName, phi)
        rmin, rmax, zmin, zmax = wall["minr"], wall["maxr"], wall["minz"], wall["maxz"]
        if box is not None:
            rmin, rmax = max(rmin, box[0]), min(rmax, box[1])
            zmin, zmax = max(zmin, box[2]), min(zmax, box[3])
        if rmax <= rmin or zmax <= zmin:  # the emission does not reach this plane
            rmin, rmax, zmin, zmax = wall["minr"], wall["minr"] + 4 * h, wall["minz"], wall["minz"] + 4 * h
        R = np.arange(rmin - h, rmax + 2 * h, h)
        z = np.arange(zmin - h, zmax + 2 * h, h)
        RR, ZZ = np.meshgrid(R, z, indexing="ij")
        emis = self.calc_emissivity(RR.ravel(), ZZ.ravel(), phi=np.array([phi]), emissionName=emissionName)[emissionName]
        grid = np.asarray(emis, dtype=float).reshape(RR.shape)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 0.0  # zero border: nothing outside the grid
        return AxisymmetricMapper(Interpolate2DLinear(R, z, grid, extrapolate=True, extrapolation_type="nearest"))

    def _camera_view_phi(self, bolo) -> tuple[float, float]:
        """
        Toroidal angle (radians) at the middle of a camera's view, and the
        toroidal span (degrees) of its central sightlines inside the wall.
        """
        phis = []
        for foil in bolo.bolometer_camera:
            origin, hit, _ = foil.trace_sightline()
            o = np.array([origin[0], origin[1], origin[2]])
            d = np.array([hit[0], hit[1], hit[2]]) - o
            pts = o[None, :] + np.linspace(0.0, 1.0, 60)[:, None] * d[None, :]
            inside = self.tokamak._inside_tokamak(np.column_stack([np.hypot(pts[:, 0], pts[:, 1]), pts[:, 2]]))
            if inside.any():
                phis.extend(np.arctan2(pts[inside, 1], pts[inside, 0]).tolist())
        if not phis:
            p = float(np.deg2rad(bolo.info["CAMERA_POSITION_R_Z_PHI"][2]))
            return p % (2.0 * np.pi), 0.0
        mid = float(np.angle(np.mean(np.exp(1j * np.asarray(phis)))))
        spread = np.angle(np.exp(1j * (np.asarray(phis) - mid)))  # relative to the middle, in (-pi, pi]
        return mid % (2.0 * np.pi), float(np.rad2deg(spread.max() - spread.min()))

    def _emission_function(self, emissionName: str, bolo, cache: dict) -> tuple[Any, str]:
        """
        What Cherab evaluates for this emission and camera: a compiled grid
        when possible (_compiled_emission), else the Python function.
        Returns (function, how) with how in "uniform", "plane", "python"
        ("grid3d" for simulation radDists).
        """
        if str(self.info.get("rayMethod", "compiled")).lower() == "python":
            return self._evaluate_cherab, "python"
        if self.TOROIDALLY_UNIFORM:
            if emissionName not in cache:
                cache[emissionName] = self._compiled_emission(emissionName, 0.0)
            return cache[emissionName], "uniform"
        phi, span = self._camera_view_phi(bolo)
        if span > self.POLOIDAL_SPAN_DEG:
            return self._evaluate_cherab, "python"  # a skewed camera sees many toroidal angles
        return self._compiled_emission(emissionName, phi), "plane"

    def observe_with_matrix(self, G) -> None:
        """
        The cameras' signals from a geometry matrix (emis3d.geometry) instead
        of ray tracing: one matrix-vector product per emission. Fills
        self.data as bolos_observe does, but each channel's signal is split
        into toroidal segments, one per sector of the grid the chord gets
        signal from, with the sector's angle in data["observed_phi_loc"]:
        the fit then applies its toroidal scale per sector and sums, which
        is exact for the grid however far the chord sweeps toroidally
        (review issue 18). A chord with no signal keeps one zero segment at
        the camera's angle. No Monte Carlo errors: the matrix's own noise is
        fixed and shared by every radDist built with it. Records the matrix
        in self.info.
        """
        if G.machine != self.info["tokamakName"]:
            raise ValueError(f"geometry matrix is for {G.machine}, not {self.info['tokamakName']}")
        units = self.info["units"]
        self.data["units"] = {"Power": "Power [W]", "Radiance": "Radiance [W / (m2 sr)]", "Brightness": "Brightness [W / m2]"}[units]
        self.data[units] = {"channelOrder": {}}
        self.data[f"{units}_error"] = {}
        cameras = {b.name: b for b in self.tokamak.bolometers if b.name in G.cameras}
        missing = [b.name for b in self.tokamak.bolometers if b.name not in G.cameras]
        if missing:
            raise ValueError(f"geometry matrix has no rows for camera(s) {missing}; rebuild it with them (emis3d build-matrix)")
        phi_sectors = G.grid.phi_centers
        self.data["observed_phi_loc"] = {}
        for emissionName in self.info["emissionNames"]:
            P = G.signals_by_sector(G.emissivity_on_grid(self, emissionName), units)  # (channels, sectors)
            self.data[units][emissionName] = {}
            self.data[f"{units}_error"][emissionName] = {}
            self.data["observed_phi_loc"][emissionName] = {}
            for ci, cam in enumerate(G.cameras):
                if cam not in cameras:
                    continue
                rows = np.flatnonzero(G.camera_of_row == ci)
                cam_phi = float(np.deg2rad(cameras[cam].info["CAMERA_POSITION_R_Z_PHI"][2])) % (2.0 * np.pi)
                segs, segs_phi = [], []
                for k in rows:
                    lit = np.flatnonzero(P[k] != 0.0)
                    if lit.size:
                        segs.append([float(v) for v in P[k, lit]])
                        segs_phi.append([float(v) for v in phi_sectors[lit]])
                    else:
                        segs.append([0.0])
                        segs_phi.append([cam_phi])
                self.data[units][emissionName][cam] = segs
                self.data[f"{units}_error"][emissionName][cam] = [[0.0] * len(s) for s in segs]
                self.data["observed_phi_loc"][emissionName][cam] = segs_phi
                self.data[units]["channelOrder"].setdefault(cam, [G.channels[k] for k in rows])
        self._phi_loc_from_segments = True
        self.info["geometryMatrix"] = {"machine": G.machine, "grid": G.grid.describe(), "samples": G.samples, "path": self.info.get("matrixPath")}

    def bolos_observe(self) -> None:
        """
        Observes the radiation function for each bolometer.

        Example found here:
        https://www.cherab.info/demonstrations/bolometry/observing_radiation_function.html#bolometer-observing-radiation
        """
        # Should be Power, Radiance, or Brightness
        units = self.info["units"]

        if units == "Power":
            self.data["units"] = "Power [W]"
        elif units == "Radiance":
            self.data["units"] = "Radiance [W / (m2 sr)]"
        elif units == "Brightness":
            self.data["units"] = "Brightness [W / m2]"

        # --- Initialize the data storage arrays
        boloCameras = self.tokamak.bolometers
        self.data[units] = {}
        self.data[f"{units}_error"] = {}
        self.data[units]["channelOrder"] = {}
        for emissionName in self.info["emissionNames"]:
            self.data[units][emissionName] = {}
            self.data[f"{units}_error"][emissionName] = {}

            for bolo_ in boloCameras:
                self.data[units][emissionName][bolo_.name] = []
                self.data[f"{units}_error"][emissionName][bolo_.name] = []

        # --- Assign sightline resolution, number of processors to be used
        self._update_bolometer_properties()

        # --- Calculate etendue's if asking for radiance
        if units in ["Radiance", "Brightness"]:
            # TIMER ADDITION
            # with phase_timer("bolos_observe/calc_etendues"):
            self.tokamak.calc_etendues()

        # --- Populate world with emitter, this cannot be a seperate definition!
        # unless you include the emitter.material changes in that def as well!
        if self.tokamak.wall is None:
            raise RuntimeError("Tokamak wall is not initialized. Ensure that the wall attribute is set before calling bolos_observe().")

        emitter = None
        # --- Add the emitter to the Emission Surface
        for val in self.tokamak.world.children:
            if val.name == "Emission Surface":
                emitter = val

        if emitter is None:
            raise RuntimeError("Could not find Tokamak Wall in tokamak.world.children during bolos_observe().")
        step = float(self.info.get("integrationStep", 0.1))
        compiled_cache: dict = {}
        self.info["emitterMethods"] = {}

        # --- Loop over each emission function within the radDist, then each bolometer
        for emissionName in self.info["emissionNames"]:

            # --- Add this to the data arrays
            self.emissionName = emissionName

            # --- Remove each bolometer from the world
            for bolo_ in boloCameras:
                bolo_._change_parent(value=None)

            # --- Observe with each bolometer
            for bolo_ in boloCameras:
                bolo_._change_parent(value=self.tokamak.world)
                # --- The emission Cherab evaluates for this camera (compiled grid or Python)
                fn, how = self._emission_function(emissionName, bolo_, compiled_cache)
                emitter.material = VolumeTransform(RadiationFunction(fn, step=step), emitter.transform.inverse())
                self.info["emitterMethods"][f"{emissionName}/{bolo_.name}"] = how
                # print(f"Observing with {bolo_.name}")
                # TIMER ADDITION
                _cam_t0 = __import__("time").perf_counter()
                observeVal = []
                observeVal_error = []
                ch_order = []

                if "FOIL_SLIT_ANGLE_FACTOR" in bolo_.info:
                    FOIL_SLIT_ANGLE_FACTOR = bolo_.info["FOIL_SLIT_ANGLE_FACTOR"]
                else:
                    FOIL_SLIT_ANGLE_FACTOR = 1.0

                # --- Set the units in the foil prior to observing the world
                for jj, foil in enumerate(bolo_.bolometer_camera):
                    ans = 0
                    ans_error = 1.0e3
                    try:

                        if units == "Brightness":
                            # --- The units need to be power or radiance for the sightline to observe,
                            # conversion to Brightness is done after observing
                            foil.units = "Radiance"
                        else:
                            foil.units = units

                        # --- Measure the radiation function for each foil
                        foil.observe()

                        if units in ["Radiance", "Brightness"]:
                            # sightline = foil.as_sightline()
                            # sightline.observe()
                            # ans = sightline.pipelines[0].value.mean

                            # --- Below is a calculation of the incident radiance directly,
                            # renormalising for comparison with the sightline
                            # --- Check for divide by zero values
                            if foil.sensitivity == 0 or foil.pipelines[0].value.mean == 0 or bolo_.etendues[jj] == 0:
                                ans = 0
                                ans_error = 0
                            else:
                                fractional_solid_angle = bolo_.etendues[jj] / foil.sensitivity
                                radiance = foil.pipelines[0].value.mean / fractional_solid_angle
                                radiance_error = (
                                    np.hypot(
                                        foil.pipelines[0].value.error() / foil.pipelines[0].value.mean,
                                        bolo_.etendues_error[jj] / bolo_.etendues[jj],
                                    )
                                    * radiance
                                )
                                if units == "Brightness":
                                    ans = radiance * 4.0 * np.pi
                                    ans_error = radiance_error / radiance * ans
                                elif units == "Radiance":
                                    ans = radiance
                                    ans_error = radiance_error

                        elif units == "Power":
                            ans = foil.pipelines[0].value.mean
                            ans_error = foil.pipelines[0].value.error()

                        ans = ans * FOIL_SLIT_ANGLE_FACTOR
                        ans_error = ans_error * FOIL_SLIT_ANGLE_FACTOR

                    except Exception as e:
                        # A failed observation must not become data: before, it was
                        # stored as 0 +- 1e3 and saved like a real measurement
                        raise RuntimeError(f"Observation failed for foil {foil.name} of camera {bolo_.name} " f"({emissionName}); no radDist was saved: {e}") from e

                    # --- Each channel stores a list of observation segments,
                    # one per toroidal angle in data["observed_phi_loc"]. For
                    # poloidal-fan cameras this is a single-element list; a
                    # toroidally-viewing chord will populate one entry per
                    # toroidal segment of the observation.
                    observeVal.append([ans])
                    observeVal_error.append([ans_error])
                    ch_order.append(foil.name)

                # --- Store the data
                self.data[units][emissionName][bolo_.name] = observeVal
                self.data[f"{units}_error"][emissionName][bolo_.name] = observeVal_error
                if bolo_.name not in self.data[units]["channelOrder"]:
                    self.data[units]["channelOrder"][bolo_.name] = ch_order

                logger.info(
                    "[timing] observe %s / %s: %.2f s",
                    emissionName,
                    bolo_.name,
                    __import__("time").perf_counter() - _cam_t0,
                )

                bolo_._change_parent(value=None)

            # --- Add each bolometer back to the world
            for bolo_ in boloCameras:
                bolo_._change_parent(value=self.tokamak.world)

    def emission_cloud(
        self,
        emissionNames=None,
        n_points: int = 20000,
        n_phi: int = 120,
        spacing: float = 0.02,
        toroidal_scale=None,
        seed: int = 0,
    ) -> dict:
        """
        Sample points in 3-D with probability proportional to emissivity, for
        drawing the radDist as a volumetric cloud.

        The emissivity is evaluated plane by plane on an (R, z) grid of the
        given spacing [m] at n_phi toroidal angles, summed over emissionNames
        (default: all), optionally multiplied by toroidal_scale(phi_rad) (for
        example a fitted toroidal profile). Points are drawn from the grid cells
        with probability proportional to emissivity and jittered within their
        cell. Returns x, y, z and the emissivity at each point.
        """
        if self.tokamak is None or self.tokamak.wall is None:
            raise RuntimeError("Build the tokamak before sampling the emission")
        names = list(emissionNames or self.info["emissionNames"])
        w = self.tokamak.wall
        Rg = np.arange(w["minr"], w["maxr"] + spacing, spacing)
        Zg = np.arange(w["minz"], w["maxz"] + spacing, spacing)
        RR, ZZ = np.meshgrid(Rg, Zg, indexing="ij")
        inside = self.tokamak._inside_tokamak(np.column_stack([RR.ravel(), ZZ.ravel()]))
        R_in, Z_in = RR.ravel()[inside], ZZ.ravel()[inside]
        phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

        emis = np.zeros((n_phi, R_in.size))
        for k, phi in enumerate(phis):  # one plane at a time: every class takes a single phi
            for name in names:
                emis[k] += np.asarray(self.calc_emissivity(R_in, Z_in, phi=np.array([phi]), emissionName=name)[name], dtype=float)
            if toroidal_scale is not None:
                emis[k] *= float(toroidal_scale(phi))
        emis = np.clip(np.nan_to_num(emis), 0.0, None)
        total = emis.sum()
        if total <= 0:
            return {"x": np.array([]), "y": np.array([]), "z": np.array([]), "emissivity": np.array([])}

        rng = np.random.default_rng(seed)
        idx = rng.choice(emis.size, size=n_points, p=(emis / total).ravel())
        k, j = np.unravel_index(idx, emis.shape)
        dphi = 2.0 * np.pi / n_phi
        R = R_in[j] + rng.uniform(-0.5, 0.5, n_points) * spacing
        Z = Z_in[j] + rng.uniform(-0.5, 0.5, n_points) * spacing
        P = phis[k] + rng.uniform(-0.5, 0.5, n_points) * dphi
        return {"x": R * np.cos(P), "y": R * np.sin(P), "z": Z, "emissivity": emis[k, j]}

    def plot3D(
        self,
        ax=None,
        emissionNames=None,
        cameras=None,
        n_points: int = 60000,
        toroidal_scale=None,
        show_injector: bool = True,
        show_field_lines: bool = True,
        view=(28.0, -60.0),
        cmap: str = "RdYlBu_r",  # "plasma",
        chord_colors: str = "height",
        point_size: float = 0.6,
        point_alpha: float = 0.3,
    ):
        """
        3-D view of the radDist (a point cloud whose density follows the
        emissivity) with the wall and the cameras' chords.

        emissionNames : components to draw (default all, summed)
        cameras       : camera names whose chords to draw (default all; [] for none)
        toroidal_scale: optional callable phi_rad -> factor, e.g. a fitted profile
        show_field_lines: draw the traced field lines' centre-lines (helical radDists)
        chord_colors  : "height" (above / at / below the midplane) or "camera"
        n_points, point_size, point_alpha: density and look of the cloud; more
                        points with smaller, fainter markers give a smoother cloud
        Returns the axes.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(1, 1, 1, projection="3d")

        cloud = self.emission_cloud(emissionNames, n_points=n_points, toroidal_scale=toroidal_scale)
        if cloud["x"].size:
            order = np.argsort(cloud["emissivity"])  # brightest drawn last, on top
            e = cloud["emissivity"][order]
            ax.scatter(
                cloud["x"][order],
                cloud["y"][order],
                cloud["z"][order],
                c=e,
                cmap=cmap,
                vmin=float(np.percentile(e, 5)),
                vmax=float(np.percentile(e, 99)),
                alpha=point_alpha,
                linewidths=0,
                depthshade=False,
                s=cast(Any, point_size),
            )  # matplotlib's stubs type s as int; sizes below 1 are wanted

        # --- Centre-lines of the traced field lines (helical radDists)
        if show_field_lines and getattr(self.tokamak, "fieldLines", None):
            fl = self.tokamak.fieldLines[next(iter(self.tokamak.fieldLines))]
            for name in fl.get("directionNames", []):
                line = fl[name]
                x, y, z = (np.asarray(line[k]) for k in ("x", "y", "z"))
                if x.ndim > 1:  # one row per traced field line: draw the one nearest the start point
                    x, y, z = x[0], y[0], z[0]
                ax.plot(x, y, z, color="black", lw=0.8, alpha=0.7)

        self.tokamak.plot3d_wall(ax)
        if cameras is None or len(cameras):
            self.tokamak.plot3d_chords(ax, cameras=cameras, color_by=chord_colors)

        wall = self.tokamak.wall
        if wall is None:
            raise RuntimeError("The tokamak has no wall to draw")
        if show_injector and "injectionLocation" in self.info:
            a = np.deg2rad(float(self.info["injectionLocation"]))
            R_out = wall["maxr"] + 0.25
            ax.plot([R_out * np.cos(a)], [R_out * np.sin(a)], [0.0], marker="v", color="tab:red", ms=10)
            ax.text(R_out * np.cos(a), R_out * np.sin(a), 0.15, "injector", color="tab:red", fontsize=9, ha="center")

        span = wall["maxr"] + 0.3
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_zlim(wall["minz"] - 0.1, wall["maxz"] + 0.1)
        ax.set_box_aspect((2 * span, 2 * span, wall["maxz"] - wall["minz"] + 0.2))
        ax.view_init(elev=view[0], azim=view[1])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(self._plot_title())
        return ax

    def _plot_title(self) -> str:
        name = self.info.get("distType", type(self).__name__)
        if "sourceFile" in self.info:  # simulation radDists
            return f"{name} radDist: {Path(str(self.info['sourceFile'])).stem}, timestep {self.info.get('timestep')}"
        return f"{name} radDist at R {float(self.info.get('startR', 0)):.2f}, z {float(self.info.get('startZ', 0)):.2f} m"

    def plotCrossSection(self, phi: float = 0.0, ax=None):
        """
        Contour plot of the radDist's emissivity at a given phi location.
        Returns the filled contour set, for a colorbar.
        """

        if self.tokamak.wall is None:
            raise RuntimeError("Tokamak wall is not initialized. Ensure that the wall attribute is set before calling plotCrossSection().")

        rLimits = (self.tokamak.wall["minr"], self.tokamak.wall["maxr"])
        zLimits = (self.tokamak.wall["minz"], self.tokamak.wall["maxz"])

        RZarray = Util_radDist.createRZGrid(rLimits, zLimits, num_r=200, num_z=200, wallcurve=None)
        if RZarray is None:
            raise RuntimeError("The RZarray returned with Util_radDist.callRZGridTokamak is None!")

        R = np.ascontiguousarray(RZarray[:, 0].ravel(), dtype=np.float64)
        z = np.ascontiguousarray(RZarray[:, 1].ravel(), dtype=np.float64)

        emiss = np.zeros(len(RZarray))

        for emissionName in self.info["emissionNames"]:
            ans = self.calc_emissivity(R, z, phi=np.array([phi]), emissionName=emissionName)
            emiss += ans[emissionName]

        # Create a new figure and axes if ax is None
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        R_unique = RZarray[:, 0]
        z_unique = RZarray[:, 1]
        n_levels = 50

        cf = ax.tricontourf(R_unique, z_unique, emiss, levels=n_levels)  # , cmap="CMRmap_r")
        ax.tricontour(
            R_unique,
            z_unique,
            emiss,
            levels=n_levels,
            colors="white",
            linewidths=0.4,
            alpha=0.4,
        )
        return cf

    def _plotOverview_crossSection(self, ax, boloGroup: str, plot_etendue: list | None = None) -> None:
        """
        Draws one top-row panel: the first wall, the chords of a single bolometer
        group, and a cross-section of the radDist at that group's toroidal location.
        """

        if plot_etendue is None:
            plot_etendue = []
        tok = self.tokamak

        tok._plot_first_wall(ax)
        tok._plot_bolometers(
            ax,
            boloGroupName=boloGroup,
            plot_chord_info=False,
            plot_etendue=plot_etendue,
            legend=False,
        )

        # --- Plot a cross-section of the radDist
        phi = tok.get_ave_bolometer_tor_loc(boloGroupName=boloGroup)
        if phi is not None:
            self.plotCrossSection(phi=np.deg2rad(phi), ax=ax)

    def _plotOverview_signals(self, ax, boloGroup: str, colors: list) -> None:
        """
        Draws one bottom-row panel: the observed emissivities of a single
        bolometer group, one line per emission name, sorted by channel number.
        """

        bolometers = self.tokamak.bolometers

        for qq, emissionName in enumerate(self.info["emissionNames"]):
            # --- Group the data
            data_ = []
            chan_ = []

            for bolo in bolometers:
                if bolo.info["GROUP_NAME"] == boloGroup:
                    ch_tags = bolo.info["CHANNEL_TAGS"]
                    c_ = []
                    for ch in ch_tags:  # type: ignore
                        c_.append(int(ch[-2:]))

                    data_ += self.data[self.info["units"]][emissionName][bolo.info["NAME"]]
                    chan_ += c_

            # --- Sort the channel list in ascending order
            inds = np.array(chan_).argsort()
            ax.plot(
                np.array(chan_)[inds],
                np.array(data_)[inds],
                color=colors[qq % len(colors)],
                label=emissionName,
            )

        ax.legend()
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_ylabel(f"{self.data['units']}")
        ax.set_xlabel("channel")
        ax.set_title(boloGroup)

    def plotOverview(
        self,
        return_figure: bool = False,
        plot_etendue: list | None = None,
        groups_per_figure: int = 3,
    ):
        """
        Plots the bolometer chords with a contour overplot of the radDist in the
        top row, and the observed emissivities in the bottom row.

        The bolometer groups are split across as many figures as needed so that
        no figure holds more than groups_per_figure of them, which keeps the
        panels readable on machines with many groups. The injection location
        cross-section is drawn as the last column of the top row on every
        figure, so each one can be read on its own.

        return_figure     :: True = returns a list of the figure objects, one per
                             chunk of bolometer groups. Returns None if the
                             tokamak has no info loaded.
        plot_etendue      :: passed through to Tokamak._plot_bolometers
        groups_per_figure :: maximum bolometer groups per figure

        Within each figure the axes are created in this order:
            [cross-section per group..., injection location, signals per group...]
        """

        if plot_etendue is None:
            plot_etendue = []
        if groups_per_figure < 1:
            raise ValueError(f"groups_per_figure must be at least 1, got {groups_per_figure}")

        tok = self.tokamak

        # --- Make the emission surface transparent for accurate ray tracing
        tok._make_raysect_surface_transparent(surfaceName="Emission Surface")

        colors = ["black", "purple", "blue", "green", "orange", "red"]

        if tok.info is None:
            logger.warning("Tokamak info is not loaded, nothing to plot.")
            return None

        boloGroups = list(tok.info["Bolometer Groups"])

        # --- Split the groups into one chunk per figure
        chunks = [boloGroups[ii : ii + groups_per_figure] for ii in range(0, len(boloGroups), groups_per_figure)]

        figures = []

        for chunkNumber, chunk in enumerate(chunks):
            # --- One column per group in this chunk, plus one for the injection
            # --- location. The last chunk may be narrower than the others.
            num_columns = len(chunk) + 1
            num_rows = 2

            f = plt.figure(figsize=(3.75 * num_columns, 8))

            # --- Top row: every bolometer with radDist contour overplot
            for ii, boloGroup in enumerate(chunk):
                f_ = f.add_subplot(num_rows, num_columns, ii + 1)
                self._plotOverview_crossSection(f_, boloGroup=boloGroup, plot_etendue=plot_etendue)

            # --- Plot the injection location in the last column of the top row
            phi = self.info["injectionLocation"]
            f_ = f.add_subplot(num_rows, num_columns, num_columns)
            tok._plot_first_wall(f_)
            self.plotCrossSection(phi=np.deg2rad(phi), ax=f_)
            f_.set_title(f"Injection Location, phi = {phi} degrees")

            # --- Bottom row: the observed emissivities. The last column of this
            # --- row is left empty, mirroring the injection panel above it.
            for ii, boloGroup in enumerate(chunk):
                f_ = f.add_subplot(num_rows, num_columns, num_columns + ii + 1)
                self._plotOverview_signals(f_, boloGroup=boloGroup, colors=colors)

            if len(chunks) > 1:
                first = chunkNumber * groups_per_figure + 1
                f.suptitle(f"Bolometer groups {first}-{first + len(chunk) - 1} " f"of {len(boloGroups)}  (figure {chunkNumber + 1} of {len(chunks)})")

            f.tight_layout()
            figures.append(f)

        if return_figure:
            return figures

        plt.show()
        return None

    def _save_names(self) -> tuple[str, str]:
        """
        (folder, file) for saveRadDist. The folder is the distType followed by
        every SAVE_KEYS parameter; the file is the start position to 0.1 mm.
        """
        parts = [str(self.info["distType"])]
        for key, label in self.SAVE_KEYS:
            if self.info.get(key) is not None:
                parts.append(f"{label}_{_name_value(self.info[key])}")
        folder = "_".join(parts)
        file = f"R_{float(self.info['startR']):.4f}_z_{float(self.info['startZ']):.4f}.json"
        return folder, file

    def saveRadDist(self) -> None:
        """
        Save the radDist info and data to
        <raddists>/<saveRunsDirectoryName>/<folder>/<file> (see _save_names).
        The file is written under a temporary name and then renamed, so an
        interrupted worker never leaves a half-written radDist.
        """
        self.info["radDistFormat"] = RADDIST_FORMAT
        toSave = {
            "info": convert_arrays_to_list(self.info),
            "data": convert_arrays_to_list(self.data),
        }

        folderName, saveFileName = self._save_names()
        pathFileName = get_workspace().path("raddists", tokamak=self.info["tokamakName"]) / self.info["saveRunsDirectoryName"] / folderName
        tmpName = f".{saveFileName}.{os.getpid()}.tmp"
        save_json(toSave, pathFileName, tmpName)
        os.replace(pathFileName / tmpName, pathFileName / saveFileName)
        logger.info("radDist saved to:\n  folder  : %s\n  filename: %s", pathFileName, saveFileName)

    @abstractmethod
    def _observed_phi_loc(self, bolo_info: dict | None = None, emissionName: str | None = None) -> list:
        """
        Abstract method to be implemented by subclasses to return, for each
        channel of the bolometer, the list of toroidal angles (radians) at
        which the observation took place. One angle per observation segment;
        single-element lists for poloidal-fan cameras.
        """


class Helical(RadDist):
    """
    Helical radDist class used to produce radDist based on the magnetic
    field line at a given R and z.

    Parameters
    ----------
    startR, startz  : float     — poloidal start location in meters.
    config          : dict      — configuration dictionary.
    setFieldLine    : bool      — Wether to trace the field lines on construction.
                                  Should be True in almost all cases
    """

    def _emission_box(self, emissionName: str, phi: float):
        R0, z0 = self.tokamak.find_RZ_Fline(str(self.info["startPhi"]), emissionName, inputPhis=np.array([phi]))
        pad = 6.0 * float(self.info["sigmaKernel"])
        return float(np.min(R0)) - pad, float(np.max(R0)) + pad, float(np.min(z0)) - pad, float(np.max(z0)) + pad

    SAVE_KEYS = (
        ("sigma_R", "sigma_R"),
        ("sigmaKernel", "kernel"),
        ("numTransists", "transits"),
        ("numFieldLines", "lines"),
        ("startPhi", "phi"),
    )

    def __init__(
        self,
        startR: float = 0.0,
        startZ: float = 0.0,
        config: dict | None = None,
        setFieldLine: bool = True,
    ) -> None:

        if config is None:
            config = {}
        super(Helical, self).__init__(startR=startR, startZ=startZ, config=config or {})

        self.info["setFieldLine"] = setFieldLine
        self.info["distType"] = "helical"

        # --- Create the field line to trace
        if setFieldLine:

            logger.info(
                "Building %s radDist | sigma_R=%.2f, R=%.2f m, z=%.2f m",
                self.info["distType"],
                self.info["sigma_R"],
                startR,
                startZ,
            )

            self._build_tokamak(
                tokamakName=self.info["tokamakName"],
                mode="Build",
                reflections=False,
                eqFileName=self.info["eqFileName"],
            )
            self.setFieldLine()

    def setFieldLine(self) -> None:
        """
        Sample source points from the bivariate normal convolution and
        trace the corresponding field lines through the tokamak.
        """

        R0 = self.info["startR"]
        z0 = self.info["startZ"]
        start_phi = self.info["startPhiRad"]
        sigma_kernel = self.info["sigmaKernel"]
        num_lines = self.info["numFieldLines"]

        # sigma_target must come from config; fall back to a small offset only
        # if not provided
        sigma_target = self.info.get("sigma_R", sigma_kernel + 0.01)
        self.info["sigma_target"] = sigma_target

        points, weights = Util_radDist.bivariate_normal_isodensity_points(R0, z0, sigma_target, sigma_kernel, num_lines, seed=42)

        self.info["weights"] = weights

        num_transits = float(self.info.get("numTransists", 1.0))
        self.tokamak.set_fieldlines(
            startR=points[:, 0],
            startZ=points[:, 1],
            startPhi=start_phi,
            numTransists=num_transits,
        )

        startPhideg = fieldline_key(start_phi, degrees=False)

        if startPhideg not in self.tokamak.fieldLines:
            raise RuntimeError(f"startPhiRad={start_phi:.4f} rad ({startPhideg}°) not found in " f"traced field lines. Available: " f"{self.tokamak.get_fieldLines_startPhis()}")

        self.info["emissionNames"] = self.tokamak.fieldLines[startPhideg]["directionNames"]
        self.info["numTransists"] = num_transits

    def _evaluate(
        self,
        R: np.ndarray,
        z: np.ndarray,
        phi: np.ndarray,
        emissionName: str | None = None,
    ) -> dict:
        """
        Return the emissivity (W/m^3/rad) at the point (R,z,ph).

        Parameters
        ----------
        R, z, phi    : array-like — evaluation coordinates.
        emissionName : str, optional — defaults to self.emissionName if not given.
        """

        # --- Set emissionName, this happens when self._evaluate_cherab() is used
        if emissionName is None:
            emissionName = self.emissionName

        R = np.ascontiguousarray(np.ravel(R), dtype=np.float64)
        z = np.ascontiguousarray(np.ravel(z), dtype=np.float64)
        phi = np.atleast_1d(np.asarray(phi, dtype=np.float64)).ravel()
        if phi.size not in (1, R.size):
            raise ValueError(f"phi must be one angle or one per point, got {phi.size} for {R.size} points")
        weights = np.asarray(self.info["weights"], dtype=np.float64)
        sigma = float(self.info["sigmaKernel"])

        # Field lines' (R, z) at each distinct angle: shape (field lines, angles)
        angles, which = np.unique(phi, return_inverse=True)
        R0, z0 = self.tokamak.find_RZ_Fline(str(self.info["startPhi"]), emissionName, inputPhis=angles)

        # Each point sums over the field lines at its own angle. (Flattening all
        # angles together used to mix field lines and angles when several were given.)
        tot = np.empty(R.size)
        for k in range(angles.size):
            sel = which == k if angles.size > 1 else slice(None)
            tot[sel] = Util_radDist._evaluate_kernels(
                R[sel],
                z[sel],
                np.ascontiguousarray(R0[:, k], dtype=np.float64),
                np.ascontiguousarray(z0[:, k], dtype=np.float64),
                weights,
                sigma,
            )
        return {emissionName: tot}

    def _observed_phi_loc(self, bolo_info: dict | None = None, emissionName: str | None = None) -> list:
        """
        Return, per channel, the list of toroidal angles (radians) at which the
        observation took place. Poloidal-fan cameras view a single toroidal
        plane, so each channel holds a single-element list at the camera phi.

        Parameters
        ----------
        bolo_info    : dict — bolometer configuration.
        emissionName : str  — used to extract the revolution number from the name.
        """

        if bolo_info is None:
            bolo_info = {}
        num_channels = bolo_info["NUM_CHANNELS"]
        phi = np.deg2rad(float(bolo_info["CAMERA_POSITION_R_Z_PHI"][2]))

        return [[phi % (2.0 * np.pi)] for _ in range(num_channels)]


class HelicalRing(RadDist):
    """
    HelicalRing radDist class used to produce radDist based on the magnetic
    field line at a given R and z.

    Parameters
    ----------
    startR, startz  : float     — centre of the field line.
    config          : dict      — dict of the configuration file.
    setFieldLine    : bool      — Trace the field line, should be true 99% of the time.
    """

    def _emission_box(self, emissionName: str, phi: float):
        flR, flZ = self.tokamak.find_RZ_Fline(str(self.info["startPhi"]), emissionName, inputPhis=np.array([phi]))
        pad = 8.0 * max(float(self.info["sigma_z"]), float(self.info["sigma_R"]) * float(self.info["sigma_z"]))
        return float(np.min(flR)) - pad, float(np.max(flR)) + pad, float(np.min(flZ)) - pad, float(np.max(flZ)) + pad

    SAVE_KEYS = (
        ("sigma_R", "sigma_R"),
        ("sigma_z", "sigma_z"),
        ("numTransists", "transits"),
        ("startPhi", "phi"),
    )

    def __init__(
        self,
        startR: float = 0.0,
        startZ: float = 0.0,
        config: dict | None = None,
        setFieldLine: bool = True,
    ):

        if config is None:
            config = {}
        super(HelicalRing, self).__init__(startR=startR, startZ=startZ, config=config or {})

        self.info["setFieldLine"] = setFieldLine
        self.info["distType"] = "helicalring"

        # --- Create the field line to trace
        if setFieldLine:
            str_ = f"→ Building HelicalRing radDist using a sigma_R of {self.info['sigma_R']:.2f} sigma_z of {self.info['sigma_z']:.2f},"
            str_ += f" starting at R = {startR:.2f}m and z = {startZ:.2f}m"
            logger.info("%s", str_)
            self._build_tokamak(
                tokamakName=self.info["tokamakName"],
                mode="Build",
                reflections=False,
                eqFileName=self.info["eqFileName"],
            )
            self.setFieldLine()

    def setFieldLine(self) -> None:
        """
        Traces the field line based on the startR and startZ
        """
        numTransists = float(self.info.get("numTransists", 1.0))

        self.tokamak.set_fieldlines(
            startR=[self.info["startR"]],
            startZ=[self.info["startZ"]],
            startPhi=self.info["startPhiRad"],
            numTransists=numTransists,
        )
        startPhideg = fieldline_key(self.info["startPhiRad"], degrees=False)

        if startPhideg not in self.tokamak.fieldLines:
            raise RuntimeError(f"Input fieldLinePhi of {startPhideg}, not availble!" f"Possible fieldLinePhi(s): {self.tokamak.get_fieldLines_startPhis()}")
        self.info["emissionNames"] = self.tokamak.fieldLines[startPhideg]["directionNames"]
        self.info["numTransists"] = numTransists

    def _evaluate(
        self,
        R: np.ndarray,
        z: np.ndarray,
        phi: np.ndarray,
        emissionName: str | None = None,
    ) -> dict:
        """
        Return the emissivity (W/m^3/rad) at the point (R,z,ph).

        Parameters
        ----------
        R, z, phi    : array-like — evaluation coordinates.
        emissionName : str, optional — defaults to self.emissionName if not given.
        """

        # --- Set emissionName if called with self._evaluate_cherab()
        if emissionName is None:
            emissionName = self.emissionName

        localEmis = {}

        vertExtendParam = 3.0  # for vertical extension of plasma... hardcoded for now

        # next we need the R,Z position of our helical structure at this phi
        flR, flZ = self.tokamak.find_RZ_Fline(str(self.info["startPhi"]), emissionName, inputPhis=phi)

        # now for bivariate normal distribution in poloidal plane.
        # elongated in approximate poloidal direction of field line

        # first we need to decompose (R,Z) in terms of parallel/perpendicular
        # to approximate field line. Approximated as the perpendicular direction
        # to the vector from (major radius, zoffset) to (flR, flZ)
        # "cent0" = (major radius, zoffset), "cent1" = (flR, flZ), "point" = (R,Z)
        if self.tokamak.info is not None:
            cent0ToCent1Vec = [flR - self.tokamak.info["MACHINE"]["majorRadius"], flZ]
            cent0ToCent1Vec[1] = cent0ToCent1Vec[1] / vertExtendParam
            cent0ToCent1VecMag = np.sqrt(cent0ToCent1Vec[0] ** 2 + cent0ToCent1Vec[1] ** 2)
            cent0ToCent1VecNormed = [x / cent0ToCent1VecMag for x in cent0ToCent1Vec]
            perpVecNormed = [-cent0ToCent1VecNormed[1], cent0ToCent1VecNormed[0]]
            cent1ToPointVec = [R - flR, z - flZ]
            paralleldist = cent1ToPointVec[0] * cent0ToCent1VecNormed[0] + cent1ToPointVec[1] * cent0ToCent1VecNormed[1]
            perpdist = cent1ToPointVec[0] * perpVecNormed[0] + cent1ToPointVec[1] * perpVecNormed[1]

            emis = (
                (1.0 / (2.0 * np.pi * self.info["sigma_R"] * (self.info["sigma_z"] ** 2)))
                * np.exp(-0.5 * (perpdist**2) / (self.info["sigma_z"] * self.info["sigma_R"]) ** 2)
                * np.exp(-0.5 * (paralleldist**2) / self.info["sigma_z"] ** 2)
            )

            localEmis[emissionName] = emis.flatten()

        return localEmis

    def _observed_phi_loc(self, bolo_info: dict | None = None, emissionName: str | None = None) -> list:
        """
        Return, per channel, the list of toroidal angles (radians) at which the
        observation took place. Poloidal-fan cameras view a single toroidal
        plane, so each channel holds a single-element list at the camera phi.

        Parameters
        ----------
        bolo_info    : dict — bolometer configuration.
        emissionName : str  — used to extract the revolution number from the name.
        """

        if bolo_info is None:
            bolo_info = {}
        num_channels = bolo_info["NUM_CHANNELS"]
        phi = np.deg2rad(float(bolo_info["CAMERA_POSITION_R_Z_PHI"][2]))

        return [[phi % (2.0 * np.pi)] for _ in range(num_channels)]


class ElongatedRing(RadDist):
    """
    Elongated Ring radDist class used to produce radDist based on the input
    R, z, sigma_R, and sigma_z.

    INPUTS:

    numBins :: The number of toroidal bins
    """

    SAVE_KEYS = (("sigma_R", "sigma_R"), ("sigma_z", "sigma_z"), ("rotationAngle", "rot"))
    TOROIDALLY_UNIFORM = True

    def _emission_box(self, emissionName: str, phi: float):
        pad = 6.0 * max(float(self.info["sigma_R"]), float(self.info["sigma_z"]))  # any rotation fits
        R0, z0 = float(self.info["startR"]), float(self.info["startZ"])
        return R0 - pad, R0 + pad, z0 - pad, z0 + pad

    def __init__(
        self,
        startR=None,
        startZ=None,
        config=None,
    ):
        # Ensure startR and startZ are floats, not None
        if config is None:
            config = {}
        if startR is None:
            startR = float(config["startR"])
        if startZ is None:
            startZ = float(config["startZ"])

        super(ElongatedRing, self).__init__(startR=startR, startZ=startZ, config=config or {})
        self.info["distType"] = "elongatedRing"
        self.info["emissionNames"] = ["elongatedRing"]

        if "sigma_R" in self.info:
            self._build_tokamak(
                tokamakName=self.info["tokamakName"],
                mode="Build",
                reflections=False,
                eqFileName=self.info["eqFileName"],
            )

            str_ = f"→ Building Elongated Ring radDist using a sigma_R of {self.info['sigma_R']:.2f}"
            str_ += f", sigma_z of {self.info['sigma_z']:.2f}, rotation angle of {self.info['rotationAngle']:.2f}, starting at R = {startR:.2f}m and z = {startZ:.2f}"

            logger.info("%s", str_)

    def _evaluate(
        self,
        R: np.ndarray,
        z: np.ndarray,
        phi: np.ndarray,
        emissionName: str | None = None,
    ) -> dict:
        """
        Return the emissivity (W/m^3/rad) at the point (R,z,ph).

        Parameters
        ----------
        R, z, phi    : array-like — evaluation coordinates.
        emissionName : str, optional — defaults to self.emissionName if not given.
        """
        # --- Set emissionName if called with self._evaluate_cherab()
        if emissionName is None:
            emissionName = self.emissionName

        localEmis = {}
        # bivariate normal distribution in poloidal plane.
        # integrated over dR and dZ this function returns 1. I think.
        localEmis[emissionName] = Util_radDist.bivariate_normal_elongated(
            R=R,
            R0=self.info["startR"],
            z=z,
            z0=self.info["startZ"],
            sigma_R=self.info["sigma_R"],
            sigma_z=self.info["sigma_z"],
            theta=self.info["rotationAngle"],
        )
        return localEmis

    def _observed_phi_loc(self, bolo_info: dict | None = None, emissionName: str | None = None) -> list:
        """
        Return, per channel, the list of toroidal angles (radians) at which the
        observation took place. Poloidal-fan cameras view a single toroidal
        plane, so each channel holds a single-element list at the camera phi.

        Parameters
        ----------
        bolo_info    : dict — bolometer configuration.
        emissionName : str  — used to extract the revolution number from the name.
        """

        if bolo_info is None:
            bolo_info = {}
        numChan = bolo_info["NUM_CHANNELS"]
        phi = np.deg2rad(float(bolo_info["CAMERA_POSITION_R_Z_PHI"][2]))

        return [[float(phi)] for _ in range(numChan)]


class SquareTube(RadDist):
    """
    Produces a square tube around the torus. This class is designed to test the
    foil.observe() functionality with a simple geometry and make sure that things are correct.

    """

    SAVE_KEYS = (("dR", "dR"), ("dz", "dz"))
    TOROIDALLY_UNIFORM = True

    def _grid_spacing(self) -> float:
        """1 mm by default: the tube has sharp edges, which a grid blurs over one cell."""
        return float(self.info.get("gridSpacing") or 1e-3)

    def _emission_box(self, emissionName: str, phi: float):
        h = 2.0 * self._grid_spacing()
        R0, z0 = float(self.info["startR"]), float(self.info["startZ"])
        dR, dz = float(self.info["dR"]), float(self.info["dz"])  # half-widths: the tube spans R0 +- dR, z0 +- dz
        return R0 - dR - h, R0 + dR + h, z0 - dz - h, z0 + dz + h

    def __init__(
        self,
        startR=None,
        startZ=None,
        config=None,
    ):
        # Ensure startR and startZ are floats, not None
        if config is None:
            config = {}
        if startR is None:
            startR = float(config["startR"])
        if startZ is None:
            startZ = float(config["startZ"])

        dR = config.get("dR")
        dz = config.get("dz")
        if dR is None or dz is None:
            raise ValueError("dR and dz must be provided in the config for SquareTube radDist.")

        super(SquareTube, self).__init__(startR=startR, startZ=startZ, config=config or {})
        self.info["distType"] = "squareTube"
        self.info["emissionNames"] = ["squareTube"]

        self._build_tokamak(
            tokamakName=self.info["tokamakName"],
            mode="Build",
            reflections=False,
            eqFileName=self.info["eqFileName"],
        )

        str_ = f"→ Building Square Tube radDist using starting at R = {startR:.2f} +/- {dR:.2f}m and z = {startZ:.2f} +/- {dz:.2f}m"
        logger.info("%s", str_)

    def _evaluate(
        self,
        R: np.ndarray,
        z: np.ndarray,
        phi: np.ndarray,
        emissionName: str | None = None,
    ) -> dict:
        """
        Return the emissivity (W/m^3/rad) at the point (R,z,ph).

        Parameters
        ----------
        R, z, phi    : array-like — evaluation coordinates.
        emissionName : str, optional — defaults to self.emissionName if not given.
        """

        # --- Set emissionName if called with self._evaluate_cherab()
        if emissionName is None:
            emissionName = self.emissionName

        localEmis = {}
        # --- Return 1 if within the square tube, 0 if outside
        R_in = (R >= self.info["startR"] - self.info["dR"]) & (R <= self.info["startR"] + self.info["dR"])
        z_in = (z >= self.info["startZ"] - self.info["dz"]) & (z <= self.info["startZ"] + self.info["dz"])
        loc_ = np.where(R_in & z_in)
        emission = np.zeros(len(R))
        emission[loc_] = 1.0

        localEmis[emissionName] = emission

        return localEmis

    def _observed_phi_loc(self, bolo_info: dict | None = None, emissionName: str | None = None) -> list:
        """
        Return, per channel, the list of toroidal angles (radians) at which the
        observation took place. Poloidal-fan cameras view a single toroidal
        plane, so each channel holds a single-element list at the camera phi.

        Parameters
        ----------
        bolo_info    : dict — bolometer configuration.
        emissionName : str  — used to extract the revolution number from the name.
        """

        if bolo_info is None:
            bolo_info = {}
        numChan = bolo_info["NUM_CHANNELS"]
        phi = np.deg2rad(float(bolo_info["CAMERA_POSITION_R_Z_PHI"][2]))

        return [[float(phi)] for _ in range(numChan)]


class FluxShell(RadDist):
    """
    Emission in a shell of flux surfaces: a Gaussian in normalized flux
    psi_N, centred on psi_center with width psi_width, toroidally uniform
    (the fit applies the toroidal shape). A radiating edge mantle is
    psi_center near 1; a hollow profile is a shell. Optionally asymmetric
    poloidally: multiplied by max(1 + asym_amp cos(theta - asym_angle), 0)
    with theta the poloidal angle about the magnetic axis (degrees, 0 on
    the outboard midplane, 90 at the top), so one side of the shell can be
    brighter (a crescent). Emission is cut off beyond psi_max (default 1.2)
    and outside the wall. Needs an equilibrium (eqFileName) for psi_N.
    """

    SAVE_KEYS = (("psi_center", "psi"), ("psi_width", "width"), ("asym_amp", "asym"), ("asym_angle", "at"))
    TOROIDALLY_UNIFORM = True

    def __init__(self, startR=None, startZ=None, config=None):
        config = {} if config is None else config
        super().__init__(startR=float(config.get("startR", 0.0) if startR is None else startR),
                         startZ=float(config.get("startZ", 0.0) if startZ is None else startZ), config=config)
        self.info["distType"] = "fluxShell"
        self.info["emissionNames"] = ["fluxShell"]
        self.info.setdefault("psi_max", 1.2)
        self.info.setdefault("asym_amp", 0.0)
        self.info.setdefault("asym_angle", 0.0)
        if "psi_center" in self.info:
            if not self.info.get("eqFileName"):
                raise ValueError("FluxShell needs an equilibrium (eqFileName) for the normalized flux")
            self._build_tokamak(tokamakName=self.info["tokamakName"], mode="Build", reflections=False, eqFileName=self.info["eqFileName"])
            if self.tokamak.gfile is None:
                raise RuntimeError(f"equilibrium {self.info['eqFileName']} did not load for {self.info['tokamakName']}")
            # The axis is the natural start position (used in save names and plots)
            self.info["startR"], self.info["startZ"] = float(self.tokamak.gfile.rmagx), float(self.tokamak.gfile.zmagx)
            logger.info("→ Building FluxShell radDist at psi_N %.3f, width %.3f", self.info["psi_center"], self.info["psi_width"])

    def psi_norm(self, R, z) -> np.ndarray:
        """Normalized flux at (R, z); large outside the equilibrium's grid (no emission there)."""
        if not hasattr(self, "_psi_interp"):
            from scipy.interpolate import RegularGridInterpolator

            g = self.tokamak.gfile
            if g is None:
                raise RuntimeError("FluxShell needs a loaded equilibrium")
            Rg = np.linspace(g.rleft, g.rleft + g.rdim, g.nx)
            Zg = np.linspace(g.zmid - g.zdim / 2, g.zmid + g.zdim / 2, g.ny)
            psin = (np.asarray(g.psi, dtype=float) - g.simagx) / (g.sibdry - g.simagx)
            self._psi_interp = RegularGridInterpolator((Rg, Zg), psin, bounds_error=False, fill_value=1e6)
        pts = np.column_stack([np.ravel(R), np.ravel(z)])
        return np.asarray(self._psi_interp(pts), dtype=float)

    def _evaluate(self, R, z, phi, emissionName=None):
        if emissionName is None:
            emissionName = self.emissionName
        psin = self.psi_norm(R, z)
        c, w = float(self.info["psi_center"]), float(self.info["psi_width"])
        emis = np.exp(-0.5 * ((psin - c) / w) ** 2)
        emis[psin > float(self.info["psi_max"])] = 0.0
        A = float(self.info.get("asym_amp", 0.0))
        if A != 0.0:
            theta = np.arctan2(np.ravel(z) - float(self.info["startZ"]), np.ravel(R) - float(self.info["startR"]))
            emis = emis * np.maximum(1.0 + A * np.cos(theta - np.deg2rad(float(self.info["asym_angle"]))), 0.0)
        return {emissionName: emis}

    def _scaling_factor(self, bolo_info=None, emissionName=None):
        return [1.0] * len(bolo_info["CHANNEL_TAGS"]) if bolo_info else []

    def _observed_phi_loc(self, bolo_info=None, emissionName=None):
        if not bolo_info:
            return []
        phi = float(np.deg2rad(bolo_info["CAMERA_POSITION_R_Z_PHI"][2])) % (2.0 * np.pi)
        return [[phi] for _ in bolo_info["CHANNEL_TAGS"]]


class FieldAlignedSource(RadDist):
    """
    A helical radiation structure from a field-line mapping
    (emis3d.fieldline_map): a Gaussian source on the injector's poloidal
    plane (startR, startZ, sigma_R, sigma_z, peak 1) carried along the
    field lines in both directions with a decay exp(-decay dphi^2). Its
    emissivity lives on the mapping's voxel grid (the geometry matrix's);
    elsewhere it is interpolated, as a simulation radDist's is. Settings:
    mapPath (the mapping file), decay, the source's shape, and phi_source
    [deg]: the source plane's toroidal angle (default the mapping's, the
    injector's); the mapped structure is rotated there (FieldLineMap.rotate).
    """

    SAVE_KEYS = (("sigma_R", "sigma_R"), ("sigma_z", "sigma_z"), ("decay", "decay"), ("phi_source", "phi"))

    def __init__(self, startR=None, startZ=None, config=None):
        config = {} if config is None else config
        super().__init__(startR=float(config.get("startR", 0.0) if startR is None else startR),
                         startZ=float(config.get("startZ", 0.0) if startZ is None else startZ), config=config)
        self.info["distType"] = "fieldAligned"
        self.info["emissionNames"] = ["fieldAligned"]
        self.info.setdefault("decay", 0.0)
        self._grid_eps = None
        if "mapPath" in self.info:
            from emis3d.fieldline_map import cached_map

            self.fmap = cached_map(self.info["mapPath"])  # the finest: plots and the saved radDist use it
            self.fmaps = {m.n_voxels: m for m in (cached_map(p) for p in self.info.get("mapPaths", [self.info["mapPath"]]))}
            self._build_tokamak(tokamakName=self.info["tokamakName"], mode="Build", reflections=False, eqFileName=self.info.get("eqFileName"))
            logger.info("→ Building field-aligned radDist from the source at R=%.3f, z=%.3f", self.info["startR"], self.info["startZ"])

    def voxel_emissivity(self, G=None) -> np.ndarray:
        """The emissivity on the mapping's voxel grid (the matrix's), from the current settings."""
        f = self.fmap if G is None else self.fmaps.get(G.grid.n_voxels)
        if f is None:
            raise ValueError("no field-line mapping for this geometry matrix (emis3d build-mapping, or list it in parametric.matrices)")
        src = f.gaussian_source(float(self.info["startR"]), float(self.info["startZ"]), float(self.info["sigma_R"]), float(self.info["sigma_z"]))
        shift = (np.deg2rad(float(self.info["phi_source"])) - f.phi0) if self.info.get("phi_source") is not None else 0.0
        return f.apply(src, float(self.info["decay"]), shift=shift)

    def _interpolator(self):
        """3-D linear interpolation of the voxel emissivity (periodic in phi), rebuilt when the settings change."""
        key = tuple(float(self.info[k]) for k in ("startR", "startZ", "sigma_R", "sigma_z", "decay")) + (float(self.info.get("phi_source") or -1e9),)
        if self._grid_eps is None or self._grid_eps[0] != key:
            from scipy.interpolate import RegularGridInterpolator

            f = self.fmap
            n_r, n_z = f.r_centers.size, f.z_centers.size
            eps = self.voxel_emissivity().reshape(n_r, f.n_phi, n_z)
            phi = np.concatenate([f.phi_centers, [f.phi_centers[0] + 2.0 * np.pi]])  # periodic
            vals = np.concatenate([eps, eps[:, :1, :]], axis=1)
            interp = RegularGridInterpolator((f.r_centers, phi, f.z_centers), vals, bounds_error=False, fill_value=0.0)
            self._grid_eps = (key, interp)
        return self._grid_eps[1]

    def _evaluate(self, R, z, phi, emissionName=None):
        if emissionName is None:
            emissionName = self.emissionName
        f = self.fmap
        R, z = np.ravel(R), np.ravel(z)
        phi_w = (np.ravel(np.broadcast_to(phi, R.shape)) - f.phi_centers[0]) % (2.0 * np.pi) + f.phi_centers[0]
        return {emissionName: np.asarray(self._interpolator()(np.column_stack([R, phi_w, z])), dtype=float)}

    def _scaling_factor(self, bolo_info=None, emissionName=None):
        return [1.0] * len(bolo_info["CHANNEL_TAGS"]) if bolo_info else []

    def _observed_phi_loc(self, bolo_info=None, emissionName=None):
        if not bolo_info:
            return []
        phi = float(np.deg2rad(bolo_info["CAMERA_POSITION_R_Z_PHI"][2])) % (2.0 * np.pi)
        return [[phi] for _ in bolo_info["CHANNEL_TAGS"]]


class SimulationRadDist(RadDist):
    """
    Emissivity from one timestep of a 3-D simulation, on a regular (R, z, phi)
    grid stored as JSON with keys pradArray, r_list, z_list and phi_list.

    Subclasses set NAME and AXES, the order of pradArray's axes.

    The toroidal grid is treated as periodic. Simulations usually write phi on
    [0, 2*pi) without the endpoint; the first plane is repeated at +2*pi so
    the interpolator also covers the last interval, and every angle is wrapped
    into the grid's period. Without this, any chord crossing the gap raised
    inside Cherab, and bolos_observe stored the whole channel as 0.

    The previous Emis3D implementation included a rotation angle in case the
    simulation used a different angle convention from Emis3D. It could be
    added here if needed.
    """

    NAME = "SIMULATION"
    AXES: tuple[str, str, str] = ("z", "r", "phi")

    def __init__(self, simFile, timestep, config=None):
        super().__init__(config=config or {})
        self.info["distType"] = self.NAME
        self.info["emissionNames"] = [self.NAME]
        self.info["sourceFile"] = simFile
        self.info["timestep"] = timestep

        self._build_tokamak(
            tokamakName=self.info["tokamakName"],
            mode="Build",
            reflections=False,
            eqFileName=self.info["eqFileName"],
        )

        logger.info("%s", f"→ Building {self.NAME} radDist")

        self._load_grid(simFile)

    def _load_grid(self, path) -> None:
        with open(path) as file:
            properties = json.load(file)

        axes = {
            "r": np.asarray(properties["r_list"], dtype=float),
            "z": np.asarray(properties["z_list"], dtype=float),
            "phi": np.asarray(properties["phi_list"], dtype=float),
        }
        values = np.asarray(properties["pradArray"], dtype=float)
        expected = tuple(axes[a].size for a in self.AXES)
        if values.shape != expected:
            raise ValueError(f"{self.NAME} pradArray has shape {values.shape}, but the axis lists give " f"{expected} for axis order {self.AXES}")
        for name, ax in axes.items():
            if ax.size < 2 or np.any(np.diff(ax) <= 0):
                raise ValueError(f"{self.NAME} {name}_list must be strictly increasing")

        # --- Periodic toroidal grid: close it with the first plane at +2*pi
        phi = axes["phi"]
        two_pi = 2.0 * np.pi
        span = phi[-1] - phi[0]
        step = float(np.median(np.diff(phi)))
        if span > two_pi + 1e-9:
            raise ValueError(f"{self.NAME} phi_list spans {np.rad2deg(span):.1f} deg, more than a full turn")
        if not np.isclose(span, two_pi, atol=1e-6):
            if not np.isclose(span + step, two_pi, rtol=0.0, atol=0.5 * step):
                raise ValueError(f"{self.NAME} phi_list covers {np.rad2deg(span):.1f} deg; a full turn is " "needed (either [0, 2*pi) or [0, 2*pi])")
            k = self.AXES.index("phi")
            values = np.concatenate([values, np.take(values, [0], axis=k)], axis=k)
            axes["phi"] = np.append(phi, phi[0] + two_pi)

        self.r_list, self.z_list, self.phi_list = axes["r"], axes["z"], axes["phi"]
        self.emis_values = values
        self.phi0 = float(axes["phi"][0])
        self.rmin, self.rmax = float(axes["r"][0]), float(axes["r"][-1])
        self.zmin, self.zmax = float(axes["z"][0]), float(axes["z"][-1])
        self.interp_operator = Interpolate3DLinear(*(axes[a] for a in self.AXES), values)

    def _compiled_emission(self, emissionName: str, phi: float = 0.0):
        """
        The simulation's own grid as a compiled Cherab function of (x, y, z):
        reordered to (R, phi, z), extended by a period so it covers Cherab's
        angle range (-pi, pi] whatever the first angle, and given a zero
        border in R and z (outside the simulation's domain the emission is
        zero, as in _evaluate, ramping down over one grid cell). Grid points
        outside the wall are zero, as calc_emissivity masks them.
        """
        two_pi = 2.0 * np.pi
        v = np.transpose(self.emis_values, [self.AXES.index(a) for a in ("r", "phi", "z")])
        phi_ax = np.asarray(self.phi_list, dtype=float)  # closed: ends one period after it starts
        shift = two_pi * np.floor((phi_ax[0] + np.pi) / two_pi)  # first angle into [-pi, pi)
        phi_ax = phi_ax - shift
        ext_phi = np.concatenate([phi_ax[:-1] - two_pi, phi_ax])
        ext_v = np.concatenate([v[:, :-1, :], v], axis=1)
        r, z = np.asarray(self.r_list, dtype=float), np.asarray(self.z_list, dtype=float)
        # Outside the wall the emission is zero, as calc_emissivity's wall mask makes it
        # (Cherab samples the whole emitting annulus, which extends beyond the wall)
        RR, ZZ = np.meshgrid(r, z, indexing="ij")
        inside = self.tokamak._inside_tokamak(np.column_stack([RR.ravel(), ZZ.ravel()])).reshape(RR.shape)
        ext_v = ext_v * inside[:, None, :]
        dr, dz = r[1] - r[0], z[1] - z[0]
        r_pad = np.concatenate([[r[0] - dr], r, [r[-1] + dr]])
        z_pad = np.concatenate([[z[0] - dz], z, [z[-1] + dz]])
        v_pad = np.pad(ext_v, ((1, 1), (0, 0), (1, 1)))
        return CylindricalTransform(Interpolate3DLinear(r_pad, ext_phi, z_pad, v_pad, extrapolate=True, extrapolation_type="nearest"))

    def _emission_function(self, emissionName: str, bolo, cache: dict) -> tuple[Any, str]:
        if str(self.info.get("rayMethod", "compiled")).lower() == "python":
            return self._evaluate_cherab, "python"
        if "grid" not in cache:
            cache["grid"] = self._compiled_emission(emissionName)
        return cache["grid"], "grid3d"

    def _evaluate(
        self,
        R: np.ndarray,
        z: np.ndarray,
        phi: np.ndarray,
        emissionName: str | None = None,
    ) -> dict:
        """
        Return the emissivity (W/m^3/rad) at the points (R, z, phi).

        Parameters
        ----------
        R, z, phi    : array-like — evaluation coordinates; phi may be a single value.
        emissionName : str, optional — defaults to self.emissionName if not given.
        """

        # --- Set emissionName if called with self._evaluate_cherab()
        if emissionName is None:
            emissionName = self.emissionName

        R = np.atleast_1d(np.asarray(R, dtype=float))
        z = np.atleast_1d(np.asarray(z, dtype=float))
        phi = np.broadcast_to(np.atleast_1d(np.asarray(phi, dtype=float)), R.shape)
        # --- Any angle, wrapped into the (closed) periodic grid
        phi = self.phi0 + np.mod(phi - self.phi0, 2.0 * np.pi)

        inside = (R >= self.rmin) & (R <= self.rmax) & (z >= self.zmin) & (z <= self.zmax)
        emis = np.zeros(R.shape)
        coords = {"r": R, "z": z, "phi": phi}
        first, second, third = (coords[a] for a in self.AXES)
        for i in np.flatnonzero(inside):
            emis[i] = self.interp_operator(float(first[i]), float(second[i]), float(third[i]))

        return {emissionName: emis}

    def _scaling_factor(self, bolo_info: dict | None = None, emissionName: str | None = None) -> list:
        """
        Compute the toroidal scaling factor (phi offset) for a bolometer channel.

        Parameters
        ----------
        bolo_info    : dict — bolometer configuration.
        emissionName : str  — used to extract the revolution number from the name.
        """

        if bolo_info is None:
            bolo_info = {}
        numChan = bolo_info["NUM_CHANNELS"]
        phi = np.deg2rad(float(bolo_info["CAMERA_POSITION_R_Z_PHI"][2]))

        return [float(phi)] * numChan

    def _observed_phi_loc(self, bolo_info: dict | None = None, emissionName: str | None = None) -> list:
        """
        Not yet implimented! (review issue 18)
        """

        if bolo_info is None:
            bolo_info = {}
        return []

    def _save_names(self) -> tuple[str, str]:
        """(folder, file): one folder per simulation file, one file per timestep."""
        stem = Path(self.info["sourceFile"]).stem
        return f"{self.info['distType']}_{stem}", f"t_{_name_value(self.info['timestep'])}.json"


class NIMROD(SimulationRadDist):
    """
    Takes emissivity data from a timestep of a NIMROD 3D MHD simulation and
    interprets it as a radiation structure. pradArray is indexed [z][r][phi].
    """

    NAME = "NIMROD"
    AXES = ("z", "r", "phi")

    def __init__(self, nimrodFile, timestep, config=None):
        super().__init__(nimrodFile, timestep, config=config)


class M3DC1(SimulationRadDist):
    """
    Takes emissivity data from a timestep of an M3DC1 3D MHD simulation and
    interprets it as a radiation structure. pradArray is indexed [phi][r][z].
    """

    NAME = "M3DC1"
    AXES = ("phi", "r", "z")

    def __init__(self, m3dc1File, timestep, config=None):
        super().__init__(m3dc1File, timestep, config=config)


def from_info(info: dict) -> "RadDist":
    """
    Rebuild a radDist (emission included) from its saved settings, e.g. to
    plot it. Simulation radDists re-read their sourceFile at their timestep;
    helical ones re-trace their field lines.
    """
    kind = str(info.get("distType", "")).lower()
    config = dict(info)
    if kind in ("nimrod", "m3dc1"):
        cls = NIMROD if kind == "nimrod" else M3DC1
        return cls(config["sourceFile"], config["timestep"], config=config)
    classes = {"elongatedring": ElongatedRing, "squaretube": SquareTube, "helical": Helical, "helicalring": HelicalRing, "fluxshell": FluxShell,
               "fieldaligned": FieldAlignedSource}
    if kind not in classes:
        raise ValueError(f"Cannot rebuild radDist of type {info.get('distType')!r}")
    return classes[kind](startR=config.get("startR"), startZ=config.get("startZ"), config=config)


def load(path) -> "RadDist":
    """Rebuild the radDist saved in a radDist file (from_info on its settings)."""
    return from_info(load_json(str(path))["info"])
