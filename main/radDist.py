# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:12:06 2021

@author: bemst

Re-organized pretty much everything, added parallelization, and a lot more during the refactor -JLH Aug., 2025

Added new Helical class that expands with the field lines, moved old helical class to helicalRing -JLH Mar. 2026

TODO: Update powerPerBin and uncomment it from the self.build()

"""

import logging

import numpy as np
from cherab.tools.emitters import RadiationFunction
from raysect.optical import VolumeTransform  # type: ignore

import main.Util_radDist as Util_radDist

logger = logging.getLogger(__name__)
from main.Globals import EMIS3D_INPUTS_DIRECTORY
from main.Tokamak import Tokamak
from main.Util import XY_To_RPhi, convert_arrays_to_list, save_json
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from abc import ABC, abstractmethod


class RadDist(ABC):
    """Parent RadDist class."""

    def __init__(self, startR=2.96, startZ=0.0, config=None):

        self.info = dict(config) if config else {}
        self.info["startR"] = startR
        self.info["startZ"] = startZ
        if "injectionLocation" in self.info:
            self.info["startPhi"] = self.info["injectionLocation"]
            self.info["startPhiRad"] = np.deg2rad(self.info["injectionLocation"])

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
        numProcessors   :: Hard-wired to 1, observation speed increases use multiple
                           processors in other locations.
        """

        boloCameras = self.tokamak.bolometers
        pixelSamples = self.info["BOLOMETER_PROPS"]["pixelSamples"]
        numProcessors = 1  # self.info["BOLOMETER_PROPS"]["numProcessors"]

        for bolo_ in boloCameras:
            # --- Either does the top or bottom loop depending on on if there is an extra bolometerCamera layer
            if hasattr(bolo_, "bolometer_camera"):
                foils = list(bolo_.bolometer_camera.foil_detectors)
            elif hasattr(bolo_, "foil_detectors"):
                foils = list(bolo_.foil_detectors)
            else:
                logger.error(
                    "Could not update bolometer properties in RadDist._update_bolometer_properties()"
                )
                foils = []
            for foil in foils:
                foil.render_engine.processes = numProcessors
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

        boloCameras = self.tokamak.bolometers
        observed_phi_loc = {}
        for emissionName in self.info["emissionNames"]:
            temp = {}
            for bolo_ in boloCameras:
                temp[bolo_.name] = self._observed_phi_loc(
                    bolo_.info, emissionName=emissionName
                )
            observed_phi_loc[emissionName] = temp
        self.data["scaleFactor"] = observed_phi_loc

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

        # self.power_per_bin_calc()
        self.calc_radiated_power()
        self.bolos_observe()
        self._get_observed_phi_loc()
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
            raise RuntimeError(
                "Initilize the tokamak prior to calling _total_radiated_power"
            )

        # -- Full grid creation
        R_vals = np.linspace(wall["minr"], wall["maxr"], n_poloidal)
        z_vals = np.linspace(wall["minz"], wall["maxz"], n_poloidal)
        RR, ZZ = np.meshgrid(R_vals, z_vals, indexing="ij")
        R_flat = RR.ravel()
        z_flat = ZZ.ravel()

        phi_array = np.linspace(0, 2.0 * np.pi, n_phi, endpoint=False)

        # --- Poloidal integral P_pol(φ) at each toroidal location
        P_pol = np.zeros(n_phi)

        for i, phi in enumerate(phi_array):

            phi_arr = np.full(len(R_flat), phi)

            # --- Caclulate the emissivity at that toroidal location
            result = self.calc_emissivity(
                R_flat, z_flat, phi_arr, emissionName=emissionName
            )
            emiss = result[emissionName]  # (N_inside,)

            # Reshape grid for integration
            emis_grid = (emiss * R_flat).reshape(n_poloidal, n_poloidal)

            inner = simpson(emis_grid, x=z_vals, axis=1)  # (n_poloidal,)
            P_pol[i] = simpson(inner, x=R_vals)

        # ── 4. Total power assuming toroidal symmetry ─────────────────────────────
        P_total = simpson(P_pol, x=phi_array)

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
            self.data["toroidalRadiatedPower"][emissionName] = (
                self._total_radiated_power(emissionName=emissionName)
            )

    def power_per_bin_calc(
        self, Errfrac: float = 0.01, Pointsupdate: int = int(1e5)
    ) -> None:
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
            raise RuntimeError(
                "Tokamak object is not initialized. Call _build_tokamak() before power_per_bin_calc()."
            )
        if not hasattr(self.tokamak, "info") or self.tokamak.info is None:
            raise RuntimeError(
                "Tokamak.info is not initialized. Ensure _build_tokamak() sets up info correctly."
            )

        volumeperbin = self.tokamak.info["MACHINE"]["volume"] / float(numBins)
        pointsperbin = 0
        reachedprecision = 0

        while reachedprecision == 0:
            pointsperbin += Pointsupdate
            # --- Create all of the random points first
            x_, y_, z_, R_, phifirstbin_ = [], [], [], [], []
            while len(x_) < Pointsupdate:
                if self.tokamak.wall is None:
                    raise RuntimeError(
                        "Tokamak wall is not initialized. Ensure that the wall attribute is set before calling power_per_bin_calc()."
                    )
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

                    emission = self.calc_emissivity(
                        R_, z_, phi, emissionName=emissionName
                    )

                    if emissionName not in self.data["emisSqArray"]:
                        self.data["emisSqArray"][emissionName] = np.zeros(numBins)
                        self.data["emisSumArray"][emissionName] = np.zeros(numBins)
                        self.data["powerPerBin"][emissionName] = np.zeros(numBins)

                    self.data["emisSumArray"][emissionName][numbin] += emission[
                        emissionName
                    ].sum()
                    self.data["emisSqArray"][emissionName][numbin] += (
                        emission[emissionName] ** 2
                    ).sum()

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
                emisvararray[key_] = (self.data["emisSqArray"][key_] / pointsperbin) - (
                    emismeanarray[key_] ** 2
                )
                integemisarray[key_] = volumeperbin * emismeanarray[key_]

                totintegemis += np.sum(integemisarray[key_])

                integemisvararray[key_] = (
                    volumeperbin**2 * emisvararray[key_] / pointsperbin
                )
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
        if not hasattr(self, "data"):
            self.data = {}

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
            self.tokamak.calc_etendues()

        # --- Populate world with emitter, this cannot be a seperate definition!
        # unless you include the emitter.material changes in that def as well!
        if self.tokamak.wall is None:
            raise RuntimeError(
                "Tokamak wall is not initialized. Ensure that the wall attribute is set before calling bolos_observe()."
            )

        emitter = None
        # --- Add the emitter to the Emission Surface
        for val in self.tokamak.world.children:
            if val.name == "Emission Surface":
                emitter = val

        if emitter is not None:
            emittting_material = VolumeTransform(
                RadiationFunction(self._evaluate_cherab), emitter.transform.inverse()
            )
            emitter.material = emittting_material
        else:
            raise RuntimeError(
                "Could not find Tokamak Wall in tokamak.world.children during bolos_observe()."
            )

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
                # print(f"Observing with {bolo_.name}")
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

                        foil.observe()
                        if units in ["Radiance", "Brightness"]:
                            # sightline = foil.as_sightline()
                            # sightline.observe()
                            # ans = sightline.pipelines[0].value.mean

                            # --- Below is a calculation of the incident radiance directly,
                            # renormalising for comparison with the sightline
                            try:
                                # --- Check for divide by zero values
                                if (
                                    foil.sensitivity == 0
                                    or foil.pipelines[0].value.mean == 0
                                    or bolo_.etendues[jj] == 0
                                ):
                                    ans = 0
                                    ans_error = 0
                                else:
                                    fractional_solid_angle = (
                                        bolo_.etendues[jj] / foil.sensitivity
                                    )
                                    radiance = (
                                        foil.pipelines[0].value.mean
                                        / fractional_solid_angle
                                    )
                                    radiance_error = (
                                        np.hypot(
                                            foil.pipelines[0].value.error()
                                            / foil.pipelines[0].value.mean,
                                            bolo_.etendues_error[jj]
                                            / bolo_.etendues[jj],
                                        )
                                        * radiance
                                    )
                                    if units == "Brightness":
                                        ans = radiance * np.pi
                                        ans_error = radiance_error / radiance * ans
                                    elif units == "Radiance":
                                        ans = radiance
                                        ans_error = radiance_error

                            except Exception as e:
                                logger.error(
                                    "Error observing radiance/brightness: %s "
                                    "(etendue=%.4g, sensitivity=%.4g, mean=%.4g, error=%.4g)",
                                    e,
                                    bolo_.etendues[jj],
                                    foil.sensitivity,
                                    foil.pipelines[0].value.mean,
                                    foil.pipelines[0].value.error(),
                                )

                        elif units == "Power":
                            ans = foil.pipelines[0].value.mean
                            ans_error = foil.pipelines[0].value.error()

                        ans = ans * FOIL_SLIT_ANGLE_FACTOR
                        ans_error = ans_error * FOIL_SLIT_ANGLE_FACTOR

                    except Exception as e:
                        logger.error("An error occurred in bolos_observe: %s", e)
                        # print(
                        #    f"Single layer cameras currently not supported, add functionality within bolos_observe!"
                        # )

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

                bolo_._change_parent(value=None)

            # --- Add each bolometer back to the world
            for bolo_ in boloCameras:
                bolo_._change_parent(value=self.tokamak.world)

    def plotCrossSection(self, phi: float = 0.0, ax=None) -> None:
        """
        Returns a contour plot of the radDist at a given phi location
        """

        if self.tokamak.wall is None:
            raise RuntimeError(
                "Tokamak wall is not initialized. Ensure that the wall attribute is set before calling plotCrossSection()."
            )

        rLimits = (self.tokamak.wall["minr"], self.tokamak.wall["maxr"])
        zLimits = (self.tokamak.wall["minz"], self.tokamak.wall["maxz"])

        RZarray = Util_radDist.createRZGrid(
            rLimits, zLimits, num_r=200, num_z=200, wallcurve=None
        )
        if RZarray is None:
            raise RuntimeError(
                "The RZarray returned with Util_radDist.callRZGridTokamak is None!"
            )

        R = np.ascontiguousarray(RZarray[:, 0].ravel(), dtype=np.float64)
        z = np.ascontiguousarray(RZarray[:, 1].ravel(), dtype=np.float64)

        emiss = np.zeros(len(RZarray))

        for emissionName in self.info["emissionNames"]:
            ans = self.calc_emissivity(
                R, z, phi=np.array([phi]), emissionName=emissionName
            )
            emiss += ans[emissionName]

        # Create a new figure and axes if ax is None
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        R_unique = RZarray[:, 0]
        z_unique = RZarray[:, 1]
        n_levels = 50

        cf = ax.tricontourf(
            R_unique, z_unique, emiss, levels=n_levels
        )  # , cmap="CMRmap_r")
        ax.tricontour(
            R_unique,
            z_unique,
            emiss,
            levels=n_levels,
            colors="white",
            linewidths=0.4,
            alpha=0.4,
        )

    def plotOverview(self, return_figure: bool = False, plot_etendue: list = []):
        """
        Plots the bolometer chords with a contour overplot of the radDist in the top row
        Plots the observed emissivities in the bottom row

        return_figure :: True = returns the figure object

        """

        tok = self.tokamak

        # --- Make the emission surface transparent for accurate ray tracing
        tok._make_raysect_surface_transparent(surfaceName="Emission Surface")

        # --- Plot everything ---
        # Top row: every bolometer with radDist contour overplot + injection location radDist on the right

        colors = ["black", "purple", "blue", "green", "orange", "red"]
        if tok.info is not None:
            boloGroups = tok.info["Bolometer Groups"]
            bolometers = tok.bolometers

            num_columns = len(boloGroups) + 1
            num_rows = 2
            f = plt.figure(figsize=(15, 8))
            plot_count = 0

            # --- Loop over each bolometer group
            for ii, boloGroup in enumerate(boloGroups):
                plot_count += 1
                f_ = f.add_subplot(num_rows, num_columns, plot_count)
                tok._plot_first_wall(f_)
                tok._plot_bolometers(
                    f_,
                    boloGroupName=boloGroup,
                    plot_chord_info=False,
                    plot_etendue=plot_etendue,
                    legend=False,
                )

                # --- Plot a cross-section of the radDist
                phi = tok.get_ave_bolometer_tor_loc(boloGroupName=boloGroup)
                if phi is not None:
                    self.plotCrossSection(phi=np.deg2rad(phi), ax=f_)

            # --- Plot the injection location
            plot_count += 1
            phi = self.info["injectionLocation"]
            f_ = f.add_subplot(num_rows, num_columns, plot_count)
            tok._plot_first_wall(f_)
            self.plotCrossSection(phi=np.deg2rad(phi), ax=f_)
            f_.set_title(f"Injection Location, phi = {phi} degrees")

            # --- Plot the observed emissivities
            for ii, boloGroup in enumerate(boloGroups):
                plot_count += 1
                f_ = f.add_subplot(num_rows, num_columns, plot_count)

                for qq, emissionName in enumerate(self.info["emissionNames"]):
                    # --- Group the data
                    data_ = []
                    chan_ = []

                    for jj, bolo in enumerate(bolometers):
                        if bolo.info["GROUP_NAME"] == boloGroup:
                            ch_tags = bolo.info["CHANNEL_TAGS"]
                            c_ = []
                            for ch in ch_tags:  # type: ignore
                                c_.append(int(ch[-2:]))

                            data_ += self.data[self.info["units"]][emissionName][
                                bolo.info["NAME"]
                            ]
                            chan_ += c_

                    # --- Sort the channel list in ascending order
                    inds = np.array(chan_).argsort()
                    f_.plot(
                        np.array(chan_)[inds],
                        np.array(data_)[inds],
                        color=colors[qq],
                        label=emissionName,
                    )

                f_.legend()
                f_.set_ylim(0, f_.get_ylim()[1])
                f_.set_ylabel(f"{self.data['units']}")
                f_.set_xlabel("channel")
                f_.set_title(boloGroup)

            plt.tight_layout()

            if return_figure:
                return f
            else:
                plt.show()

    def _folder_suffix(self) -> str:
        """
        Return the unique suffix appended to this distribution's save folder name.
        Subclasses that need a different suffix (e.g. Helical) should override this.
        """
        return f"_rotation{self.info['rotationAngle']}"

    def saveRadDist(self) -> None:
        """Save the radDist info and data to a JSON file."""
        toSave = {
            "info": convert_arrays_to_list(self.info),
            "data": convert_arrays_to_list(self.data),
        }
        folderName = (
            f"{self.info['distType']}_sigma_R_{self.info['sigma_R']}_sigma_z_{self.info['sigma_z']}"
            f"{self._folder_suffix()}"
        )
        saveFileName = f"R_{self.info['startR']:.2f}_z_{self.info['startZ']:.2f}.json"
        pathFileName = (
            EMIS3D_INPUTS_DIRECTORY
            / self.info["tokamakName"]
            / "radDists"
            / self.info["saveRunsDirectoryName"]
            / folderName
        )
        save_json(toSave, pathFileName, saveFileName)
        str_ = (
            f"radDist saved to:\n"
            f"  folder  : {folderName}\n"
            f"  filename: {saveFileName}"
        )
        logger.info("%s", str_)

    @abstractmethod
    def _observed_phi_loc(
        self, bolo_info: dict = {}, emissionName: str | None = None
    ) -> list:
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

    def __init__(
        self,
        startR: float = 0.0,
        startZ: float = 0.0,
        config: dict = {},
        setFieldLine: bool = True,
    ) -> None:

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

    def _folder_suffix(self) -> str:
        return f"_sigmaKernel{self.info['sigmaKernel']}"

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

        points, weights = Util_radDist.bivariate_normal_isodensity_points(
            R0, z0, sigma_target, sigma_kernel, num_lines, seed=42
        )

        self.info["weights"] = weights

        self.tokamak.set_fieldlines(
            startR=points[:, 0],
            startZ=points[:, 1],
            startPhi=start_phi,
            numTransists=1.0,
        )

        startPhideg = str(int(np.rad2deg(start_phi)))

        if startPhideg not in self.tokamak.fieldLines:
            raise RuntimeError(
                f"startPhiRad={start_phi:.4f} rad ({startPhideg}°) not found in "
                f"traced field lines. Available: "
                f"{self.tokamak.get_fieldLines_startPhis()}"
            )

        self.info["emissionNames"] = self.tokamak.fieldLines[startPhideg][
            "directionNames"
        ]
        self.info["numTransists"] = 1.0

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

        R0_arr, z0_arr = self.tokamak.find_RZ_Fline(
            str(self.info["startPhi"]), emissionName, inputPhis=phi
        )

        # Squeeze out any spurious dimensions, ensure C-contiguous float64
        R = np.ascontiguousarray(R.ravel(), dtype=np.float64)
        z = np.ascontiguousarray(z.ravel(), dtype=np.float64)
        R0_arr = np.ascontiguousarray(R0_arr.ravel(), dtype=np.float64)
        z0_arr = np.ascontiguousarray(z0_arr.ravel(), dtype=np.float64)

        tot = Util_radDist._evaluate_kernels(
            R,
            z,
            R0_arr,
            z0_arr,
            self.info["weights"].astype(np.float64),
            float(self.info["sigma_R"]),
        )

        return {emissionName: tot}

    def _observed_phi_loc(
        self, bolo_info: dict = {}, emissionName: str | None = None
    ) -> list:
        """
        Return, per channel, the list of toroidal angles (radians) at which the
        observation took place. Poloidal-fan cameras view a single toroidal
        plane, so each channel holds a single-element list at the camera phi.

        Parameters
        ----------
        bolo_info    : dict — bolometer configuration.
        emissionName : str  — used to extract the revolution number from the name.
        """

        num_channels = bolo_info["NUM_CHANNELS"]
        phi = np.deg2rad(float(bolo_info["CAMERA_POSITION_R_Z_PHI"][2]))

        rev_number = 0
        if emissionName is not None and "rev" in emissionName:
            rev_number = int(emissionName.split("rev")[-1])

        return [[phi + rev_number * 2.0 * np.pi] for _ in range(num_channels)]


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

    def __init__(
        self,
        startR: float = 0.0,
        startZ: float = 0.0,
        config: dict = {},
        setFieldLine: bool = True,
    ):

        super(HelicalRing, self).__init__(
            startR=startR, startZ=startZ, config=config or {}
        )

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
        numTransists = 1.0

        self.tokamak.set_fieldlines(
            startR=[self.info["startR"]],
            startZ=[self.info["startZ"]],
            startPhi=self.info["startPhiRad"],
            numTransists=numTransists,
        )
        startPhideg = f'{int(np.rad2deg(self.info["startPhiRad"]))}'

        if startPhideg not in self.tokamak.fieldLines:
            raise RuntimeError(
                f"Input fieldLinePhi of {startPhideg}, not availble!"
                f"Possible fieldLinePhi(s): {self.tokamak.get_fieldLines_startPhis()}"
            )
        self.info["emissionNames"] = self.tokamak.fieldLines[startPhideg][
            "directionNames"
        ]
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
        R0, z0 = self.tokamak.find_RZ_Fline(
            str(self.info["startPhi"]), emissionName, inputPhis=phi
        )
        R0 = R0.flatten()
        z0 = z0.flatten()

        vertExtendParam = 3.0  # for vertical extension of plasma... hardcoded for now

        # next we need the R,Z position of our helical structure at this phi
        flR, flZ = self.tokamak.find_RZ_Fline(
            str(self.info["startPhi"]), emissionName, inputPhis=phi
        )

        # now for bivariate normal distribution in poloidal plane.
        # elongated in approximate poloidal direction of field line

        # first we need to decompose (R,Z) in terms of parallel/perpendicular
        # to approximate field line. Approximated as the perpendicular direction
        # to the vector from (major radius, zoffset) to (flR, flZ)
        # "cent0" = (major radius, zoffset), "cent1" = (flR, flZ), "point" = (R,Z)
        if self.tokamak.info is not None:
            cent0ToCent1Vec = [flR - self.tokamak.info["MACHINE"]["majorRadius"], flZ]
            cent0ToCent1Vec[1] = cent0ToCent1Vec[1] / vertExtendParam
            cent0ToCent1VecMag = np.sqrt(
                cent0ToCent1Vec[0] ** 2 + cent0ToCent1Vec[1] ** 2
            )
            cent0ToCent1VecNormed = [x / cent0ToCent1VecMag for x in cent0ToCent1Vec]
            perpVecNormed = [-cent0ToCent1VecNormed[1], cent0ToCent1VecNormed[0]]
            cent1ToPointVec = [R - flR, z - flZ]
            paralleldist = (
                cent1ToPointVec[0] * cent0ToCent1VecNormed[0]
                + cent1ToPointVec[1] * cent0ToCent1VecNormed[1]
            )
            perpdist = (
                cent1ToPointVec[0] * perpVecNormed[0]
                + cent1ToPointVec[1] * perpVecNormed[1]
            )

            emis = (
                (
                    1.0
                    / (2.0 * np.pi * self.info["sigma_R"] * (self.info["sigma_z"] ** 2))
                )
                * np.exp(
                    -0.5
                    * (perpdist**2)
                    / (self.info["sigma_z"] * self.info["sigma_R"]) ** 2
                )
                * np.exp(-0.5 * (paralleldist**2) / self.info["sigma_z"] ** 2)
            )

            localEmis[emissionName] = emis.flatten()

        return localEmis

    def _observed_phi_loc(
        self, bolo_info: dict = {}, emissionName: str | None = None
    ) -> list:
        """
        Return, per channel, the list of toroidal angles (radians) at which the
        observation took place. Poloidal-fan cameras view a single toroidal
        plane, so each channel holds a single-element list at the camera phi.

        Parameters
        ----------
        bolo_info    : dict — bolometer configuration.
        emissionName : str  — used to extract the revolution number from the name.
        """

        num_channels = bolo_info["NUM_CHANNELS"]
        phi = np.deg2rad(float(bolo_info["CAMERA_POSITION_R_Z_PHI"][2]))

        rev_number = 0
        if emissionName is not None and "rev" in emissionName:
            rev_number = int(emissionName.split("rev")[-1])

        return [[phi + rev_number * 2.0 * np.pi] for _ in range(num_channels)]


class ElongatedRing(RadDist):
    """
    Elongated Ring radDist class used to produce radDist based on the input
    R, z, sigma_R, and sigma_z.

    INPUTS:

    numBins :: The number of toroidal bins
    """

    def __init__(
        self,
        startR=None,
        startZ=None,
        config={},
    ):
        # Ensure startR and startZ are floats, not None
        if startR is None:
            startR = float(config.get("startR"))
        if startZ is None:
            startZ = float(config.get("startZ"))

        super(ElongatedRing, self).__init__(
            startR=startR, startZ=startZ, config=config or {}
        )
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

    def _observed_phi_loc(
        self, bolo_info: dict = {}, emissionName: str | None = None
    ) -> list:
        """
        Return, per channel, the list of toroidal angles (radians) at which the
        observation took place. Poloidal-fan cameras view a single toroidal
        plane, so each channel holds a single-element list at the camera phi.

        Parameters
        ----------
        bolo_info    : dict — bolometer configuration.
        emissionName : str  — used to extract the revolution number from the name.
        """

        numChan = bolo_info["NUM_CHANNELS"]
        phi = np.deg2rad(float(bolo_info["CAMERA_POSITION_R_Z_PHI"][2]))

        return [[float(phi)] for _ in range(numChan)]


class SquareTube(RadDist):
    """
    Produces a square tube around the torus. This class is designed to test the
    foil.observe() functionality with a simple geometry and make sure that things are correct.

    """

    def __init__(
        self,
        startR=None,
        startZ=None,
        config={},
    ):
        # Ensure startR and startZ are floats, not None
        if startR is None:
            startR = float(config.get("startR"))
        if startZ is None:
            startZ = float(config.get("startZ"))

        dR = config.get("dR")
        dz = config.get("dz")
        if dR is None or dz is None:
            raise ValueError(
                "dR and dz must be provided in the config for SquareTube radDist."
            )

        super(SquareTube, self).__init__(
            startR=startR, startZ=startZ, config=config or {}
        )
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
        R_in = (R >= self.info["startR"] - self.info["dR"]) & (
            R <= self.info["startR"] + self.info["dR"]
        )
        z_in = (z >= self.info["startZ"] - self.info["dz"]) & (
            z <= self.info["startZ"] + self.info["dz"]
        )
        loc_ = np.where(R_in & z_in)
        emission = np.zeros(len(R))
        emission[loc_] = 1.0

        localEmis[emissionName] = emission

        return localEmis

    def _observed_phi_loc(
        self, bolo_info: dict = {}, emissionName: str | None = None
    ) -> list:
        """
        Return, per channel, the list of toroidal angles (radians) at which the
        observation took place. Poloidal-fan cameras view a single toroidal
        plane, so each channel holds a single-element list at the camera phi.

        Parameters
        ----------
        bolo_info    : dict — bolometer configuration.
        emissionName : str  — used to extract the revolution number from the name.
        """

        numChan = bolo_info["NUM_CHANNELS"]
        phi = np.deg2rad(float(bolo_info["CAMERA_POSITION_R_Z_PHI"][2]))

        return [[float(phi)] for _ in range(numChan)]
