# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:12:06 2021

@author: bemst

Re-organized pretty much everything, added parallelization, and a lot more during the refactor -JLH Aug., 2025
"""

import os

import numpy as np
from cherab.tools.emitters import RadiationFunction
import pdb
from matplotlib import cm
from raysect.core.math import translate
from raysect.optical import VolumeTransform  # type: ignore

from raysect.primitive import Cylinder

import main.Util_radDist as Util_radDist
from main.Globals import *
from main.Tokamak import Tokamak
from main.Util import XY_To_RPhi, convert_arrays_to_list, save_json


from abc import ABC, abstractmethod


class RadDist(ABC):
    """
    Parent RadDist class.
    """

    def __init__(self, startR=2.96, startZ=0.0, config={}):

        self.info = config
        self.info["startR"] = startR
        self.info["startZ"] = startZ
        if "injectionLocation" in config:
            self.info["startPhi"] = config["injectionLocation"]
            self.info["startPhiRad"] = np.deg2rad(config["injectionLocation"])

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

    def _evalulateCherab(self, X, Y, Z) -> np.ndarray:
        """
        Wrapper function for self.evalulate to take inputs from Cherab
        """
        theta = self.info["rotationAngle"]

        R, phi = XY_To_RPhi(X, Y)
        # --- Convert phi to be positive
        if phi is not None and phi < 0:
            phi += 2.0 * np.pi
        emission = self.evaluate([R], [Z], [phi], theta, emissionName=self.emissionName)
        return emission[self.emissionName].item()

    @abstractmethod
    def evaluate(self, R, z, phi, theta, emissionName=None) -> dict:
        """
        Abstract method to be implemented by subclasses to return the emissivity at the given point.
        """
        pass

    def build(self) -> None:
        """
        Creates radDist, finds the power per bin, observes the world for
        each bolometer, then saves the data.

        The tokamak should be created by the individual radDist subtypes
        (helical, elongated ring, etc.) during startup.
        """
        self.power_per_bin_calc()
        self.bolos_observe()
        self._get_scale_factor()
        self.saveRadDist()

    def power_per_bin_calc(self, Errfrac=0.01, Pointsupdate=int(1e5)) -> None:
        """
        Replaces the power_per_bin_calc with a vectorized version.

        """

        # --- Initilize items
        self.data = {}
        self.data["emisSumArray"] = {}
        self.data["emisSqArray"] = {}
        self.data["powerPerBin"] = {}

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
            theta = self.info["rotationAngle"]

            for numbin in range(0, numBins):
                # print(f"Calculating powerPerBin {numbin + 1} of {numBins}")

                # --- use the initial point for each toroidal bin
                phi = np.array(phifirstbin_) + (angleperbin * numbin)

                # Add the emission to the existing arrays
                for emissionName in self.info["emissionNames"]:

                    # Choose between original emission and emission from Ben's ElongatedHelical Class
                    
                    emission = self.evaluate(
                        R_, z_, phi, theta, emissionName=emissionName
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
            # I believe they are checking to see if the variance between bins
            # is small? -JLH
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
                    print(
                        f"Number of points {pointsperbin}, total std. err fraction so far = {toterrfrac:.2e}"
                    )

    def _update_bolometer_properties(self) -> None:
        """
        Changes Cherab observation parameters. These should be defined within the radDist
        config file under BOLOMETER_PROPS

        Required values in config file:
        pixelSamples    :: Resolution of the sightline. Higher = better, but takes longer.
        numProcessors   :: The number of processors to use while observing
        """

        boloCameras = self.tokamak.bolometers
        pixelSamples = self.info["BOLOMETER_PROPS"]["pixelSamples"]
        numProcessors = self.info["BOLOMETER_PROPS"]["numProcessors"]

        for bolo_ in boloCameras:
            # --- Either does the top or bottom loop depending on on if there is an extra bolometerCamera layer
            if hasattr(bolo_, "bolometer_camera"):
                foils = list(bolo_.bolometer_camera.foil_detectors)
            elif hasattr(bolo_, "foil_detectors"):
                foils = list(bolo_.foil_detectors)
            else:
                print("---  ERROR. ---")
                print(
                    f"Could not update bolometer properties in RadDist._update_bolometer_properties()"
                )
                print("---  ERROR. ---")
                foils = []
            for foil in foils:
                foil.render_engine.processes = numProcessors
                foil.pixel_samples = pixelSamples

    def _get_scale_factor(self) -> None:
        """
        Creates a dict containing a nested list of the scaling factors for each synthetic signal,
        to be used in the fitting. The list has the same form as the radDist.

        It will first see to see if there is a specific radDist called _scaling_factor, otherwise
        it will returns 1's.

        Form:
        [
            [                               emissionName1
                [bolo1_1, bolo1_2, ...]
                [bolo2_1, bolo2_2, ...],
                ...
            ],
            [                               emissionName2
                [bolo1_1, bolo1_2, ...]
                [bolo2_1, bolo2_2, ...],
                ...
            ],
            ...
        ]
        """

        boloCameras = self.tokamak.bolometers
        scaleFactor = {}
        for emissionName in self.info["emissionNames"]:
            temp = {}
            for bolo_ in boloCameras:
                temp[bolo_.name] = self._scaling_factor(
                    bolo_.info, emissionName=emissionName
                )
            scaleFactor[emissionName] = temp
        self.data["scaleFactor"] = scaleFactor

    def bolos_observe(self) -> None:
        """
        Observes the radiation function for each bolometer
        """
        # Should be Power or Radiance
        units = self.info["units"]

        if units == "Power":
            self.data["units"] = "Power [W]"
        elif units == "Radiance":
            self.data["units"] = "Radiance [W / (m2 sr)]"

        # --- Initilize the data storage arrays
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

        # --- Populate world with emitter, this cannot be a seperate definition!
        # unless you include the emitter.material changes in that def as well!
        if self.tokamak.wall is None:
            raise RuntimeError(
                "Tokamak wall is not initialized. Ensure that the wall attribute is set before calling bolos_observe()."
            )
        dz = np.abs(self.tokamak.wall["maxz"]) + np.abs(self.tokamak.wall["minz"])
        dR = np.abs(self.tokamak.wall["maxz"]) + np.abs(self.tokamak.wall["minz"])
        radius = np.round(dR * 1.2, 2)
        height = np.round(dz * 1.2, 2)
        offset = -np.round(height / 2.0, 2)
        emitter = Cylinder(
            radius=radius, height=height, transform=translate(0, 0, offset)
        )
        emitter.parent = self.tokamak.world

        # --- Loop over each emission function within the radDist, then each bolometer
        for emissionName in self.info["emissionNames"]:

            # --- Add this to the data arrays
            self.emissionName = emissionName
            emitter.material = VolumeTransform(
                RadiationFunction(self._evalulateCherab), emitter.transform.inverse()
            )

            # --- Observe with each bolometer
            for bolo_ in boloCameras:
                # print(f"Observing {emissionName} for {bolo_.name}")
                observeVal = []
                observeVal_error = []
                ch_order = []

                # --- Calculate etendue's if asking for radiance
                if units == "Radiance":
                    if not hasattr(bolo_, "etendues"):
                        bolo_.calc_etendues()

                # --- Set the units in the foil prior to observing the world
                for jj, foil in enumerate(bolo_.bolometer_camera):
                    ans = 0
                    ans_error = 1.0e3
                    try:
                        foil.units = units
                        foil.observe()
                        if units == "Radiance":
                            fractional_solid_angle = (
                                bolo_.etendues[jj] / foil.sensitivity
                            )
                            ans = foil.pipelines[0].value.mean / fractional_solid_angle
                            ans_error = (
                                np.hypot(
                                    foil.pipelines[0].value.error()
                                    / foil.pipelines[0].value.mean,
                                    bolo_.etendues_error[jj] / bolo_.etendues[jj],
                                )
                                * ans
                            )
                        elif units == "Power":
                            ans = foil.pipelines[0].value.mean
                            ans_error = foil.pipelines[0].value.error()

                    except Exception as e:
                        print(f"An error occured: {e}")
                        # print(
                        #    f"Single layer cameras currently not supported, add functionality within bolos_observe!"
                        # )
                    observeVal.append(ans)
                    observeVal_error.append(ans_error)
                    ch_order.append(foil.name)

                # --- Store the data
                self.data[units][emissionName][bolo_.name] = observeVal
                self.data[f"{units}_error"][emissionName][bolo_.name] = observeVal_error
                if bolo_.name not in self.data[units]["channelOrder"]:
                    self.data[units]["channelOrder"][bolo_.name] = ch_order

    def plotCrossSection(self, phi=0, ax=None) -> None:
        """
        Returns a contour plot of the radDist at a given phi location
        """
        theta = self.info["rotationAngle"]
        if self.tokamak.wall is None:
            raise RuntimeError(
                "Tokamak wall is not initialized. Ensure that the wall attribute is set before calling plotCrossSection()."
            )
        rLimits = [self.tokamak.wall["minr"], self.tokamak.wall["maxr"]]
        zLimits = [self.tokamak.wall["minz"], self.tokamak.wall["maxz"]]
        rarray = np.linspace(rLimits[0], rLimits[1], num=1_000)
        zarray = np.linspace(zLimits[0], zLimits[1], num=1_000)

        emiss = np.zeros((rarray.shape[0], zarray.shape[0]))
        emiss_txt = np.zeros((rarray.shape[0], zarray.shape[0]))
        loc_ = []
        for emissionName in self.info["emissionNames"]:
            for ii in range(rarray.shape[0]):
                temp = self.evaluate(
                    rarray[ii],
                    zarray,
                    phi=[phi],
                    theta=theta,
                    emissionName=emissionName,
                )
                emiss[ii, :] += temp[emissionName].squeeze()
                emiss_txt[ii, :] += temp[emissionName].squeeze()

            # --- Add text at the peak location
            loc_.append(np.unravel_index(emiss_txt.argmax(), emiss_txt.shape))
            emiss_txt = np.zeros((rarray.shape[0], zarray.shape[0]))

        # Create a new figure and axes if ax is None
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        ax.contourf(rarray, zarray, emiss.transpose(), cmap=cm.get_cmap("BuGn"))

        for ii, emissionName in enumerate(self.info["emissionNames"]):
            ax.text(
                rarray[loc_[ii][0]],
                zarray[loc_[ii][1]],
                emissionName,
                zorder=1,
                ha="center",
                va="center",
                color="black",
                weight="bold",
            )

    def saveRadDist(self) -> None:
        """
        Saves the data and radDist information
        """

        # --- Convert items within the self.info and self.data to lists
        toSave = {
            "info": convert_arrays_to_list(self.info),
            "data": convert_arrays_to_list(self.data),
        }

        # --- Save the data
        folderName = f"{self.info['distType']}_elongation_{self.info['elongation']}_polSigma_{self.info['polSigma']}_rotation{self.info['rotationAngle']}"
        saveFileName = f"R_{self.info['startR']:.2f}_z_{self.info['startZ']:.2f}.json"

        pathFileName = os.path.join(
            EMIS3D_INPUTS_DIRECTORY,
            self.info["tokamakName"],
            "radDists",
            self.info["saveRunsDirectoryName"],
            folderName,
        )

        save_json(toSave, pathFileName, saveFileName)

    @abstractmethod
    def _scaling_factor(self, bolo_info, emissionName=None) -> list:
        """
        Abstract method to be implemented by subclasses to return the
        scaling factor for the bolometer.
        """


class Helical(RadDist):
    """
    Helical radDist class used to produce radDist based on the magnetic
    field line at a given R and z.

    INPUTS:

    setFieldLine :: Trace the field line, should be true 99% of the time
    """

    def __init__(
        self,
        startR=None,
        startZ=None,
        config={},
        setFieldLine=True,
    ):
        # Ensure startR and startZ are floats, not None
        if startR is None:
            startR = float(config.get("startR", 2.96))
        if startZ is None:
            startZ = float(config.get("startZ", 0.0))

        super(Helical, self).__init__(startR=startR, startZ=startZ, config=config)

        self.info["setFieldLine"] = setFieldLine
        self.info["distType"] = "helical"

        # --- Create the field line to trace
        if setFieldLine:
            str_ = f"Building Helical radDist using a polSigma of {self.info['polSigma']:.2f},"
            str_ += f" starting at R = {startR:.2f}m and z = {startZ:.2f}m"
            print(str_)
            self._build_tokamak(
                tokamakName=self.info["tokamakName"],
                mode="Build",
                reflections=False,
                eqFileName=self.info["eqFileName"],
            )
            self.setFieldLine()

    def setFieldLine(self) -> None:
        numTransists = 2.0

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

    def evaluate(self, R, z, phi, theta, emissionName=None) -> dict:
        """
        Return the emissivity (W/m^3/rad) at the point (R,z,ph) according to this
        instantiation of Emis3D.

        Inputs:
            R :: float, list
                 R locations to evalulate, meters
            z :: float, list
                 z locations to evaluate, meters
            phi :: float, list
                   phi locations of the field lines, in radians
            emissionName :: str, optional
                            The name of the emission to evalulate. If None, uses self.emissionName

        """
        # --- Set emissionName if called with self._evalulateCherab()
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
        flR, flZ = self.tokamak.find_RZ_Fline(str(self.info["startPhi"]), emissionName, inputPhis=phi)

        # now for bivariate normal distribution in poloidal plane.
        # elongated in approximate poloidal direction of field line

        # first we need to decompose (R,Z) in terms of parallel/perpendicular
        # to approximate field line. Approximated as the perpendicular direction
        # to the vector from (major radius, zoffset) to (flR, flZ)
        # "cent0" = (major radius, zoffset), "cent1" = (flR, flZ), "point" = (R,Z)    
        
        cent0ToCent1Vec = [flR - self.tokamak.info['MACHINE']['majorRadius'], flZ]
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
            (1.0 / (2.0 * np.pi * self.info["elongation"] * (self.info["polSigma"]**2)))
            * np.exp(
                -0.5 * (perpdist**2) / (self.info["polSigma"] * self.info["elongation"]) ** 2
            )
            * np.exp(-0.5 * (paralleldist**2) / self.info["polSigma"]**2)
        )
        
        localEmis[emissionName] = emis

        return localEmis

    def _scaling_factor(self, bolo_info, emissionName=None) -> list:
        """
        Returns the scaling factor for the bolometer.
        """
        numChan = bolo_info["NUM_CHANNELS"]
        phi = np.deg2rad(float(bolo_info["CAMERA_POSITION_R_Z_PHI"][2]))

        # --- Find the revolution number, this needs to be more universal...
        if emissionName is not None:
            # _rev0, rev1, rev2, ...
            if "rev" in emissionName:
                revNumber = int(emissionName.split("rev")[-1])
            else:
                revNumber = 0
        else:
            revNumber = 0

        return [float(phi) + revNumber * 2.0 * np.pi] * numChan


class ElongatedRing(RadDist):
    """
    Elongated Ring radDist class used to produce radDist based on the input
    R, z, polSigma, and elongation.

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
            startR = float(config.get("startR", 2.96))
        if startZ is None:
            startZ = float(config.get("startZ", 0.0))

        super(ElongatedRing, self).__init__(startR=startR, startZ=startZ, config=config)
        self.info["distType"] = "elongatedRing"
        self.info["emissionNames"] = ["elongatedRing"]

        if "polSigma" in self.info:
            self._build_tokamak(
                tokamakName=self.info["tokamakName"],
                mode="Build",
                reflections=False,
                eqFileName=self.info["eqFileName"],
            )

            str_ = f"Building Elongated Ring radDist using a polSigma of {self.info['polSigma']:.2f}"
            str_ += f", elongation of {self.info['elongation']:.2f}, rotation angle of {self.info['rotationAngle']:.2f}, starting at R = {startR:.2f}m and z = {startZ:.2f}"

            print(str_)

    def evaluate(self, R, z, phi, theta, emissionName=None) -> dict:
        """
        Find the emissivity given and input R, z, and phi location
        """
        # --- Set emissionName if called with self._evalulateCherab()
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
            elongation=self.info["elongation"],
            polSigma=self.info["polSigma"],
            theta=self.info["rotationAngle"],
        )
        return localEmis

    def _scaling_factor(self, bolo_info, emissionName=None) -> list:
        """
        Returns the scaling factor for the bolometer.
        """
        numChan = bolo_info["NUM_CHANNELS"]
        phi = np.deg2rad(float(bolo_info["CAMERA_POSITION_R_Z_PHI"][2]))

        return [float(phi)] * numChan
