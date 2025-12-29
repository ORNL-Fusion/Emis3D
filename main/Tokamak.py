#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:43:28 2023

@author: bsteinlu

Updated and re-written during the refactor - JLH

"""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from freeqdsk import geqdsk
from matplotlib import path
from raysect.core import Point3D
from raysect.core.math import rotate_z, translate
from raysect.optical import World
from raysect.optical.library import RoughTungsten
from raysect.optical import AbsorbingSurface  # type: ignore

from raysect.primitive import Box, Cylinder, import_stl

from main.Diagnostic import Bolometer
from main.Fieldline_Tracer import Fieldline_Tracer
from main.Globals import *
from main.Util import (
    config_loader,
    point3d_to_rz,
    draw_radial_lines,
    find_intersection,
    rz_to_xyz,
    split_revolutions,
)


class Tokamak(object):

    def __init__(
        self,
        tokamakName=None,
        mode="Analysis",
        reflections=False,
        eqFileName=None,
        loadBolometers=False,
        verbose=False,
    ):
        """
        Basic tokamak class which loads information specific to the TokamakName.

        This class loads the configuration file for the tokamak within
        ../tokamaks/{DIII-D, SPARC, etc}/{DIII-D, SPARC, etc}_settings.yaml

        The file should contain information about the SXR/bolometer arrays,
        wall file location, volume, majorRadius, minorRadius, etc.

        INPUTS:

        tokamakName :: The name of the tokamak (e.g. DIII-D, SPARC)
        mode        :: Analysis or Build
        reflections :: Boolean, determines if the tokamak reflects the radiation
        eqFileName  :: The name of the equilibrium file to be used
        loadBolometers    :: Set to True to load the bolometers (needed to make radDists)
        """
        self.verbose = verbose
        if tokamakName not in SUPPORTED_TOKAMAKS:
            print(f"Please eneter a valid tokamak name!")
            print(f"Tokamaks currently supported are: {SUPPORTED_TOKAMAKS}")
            raise Exception

        else:
            # --- Load the configuration file
            self._load_config_file(tokamakName, mode, reflections, eqFileName)

            # --- Set the general input directory
            self.input_dir = os.path.join(
                EMIS3D_TOKMAK_DIRECTORY, tokamakName, "inputs"
            )

            # --- Run the startup program
            self._tokamak_startup()

            # --- Load the bolometers
            if loadBolometers:
                self._load_bolometers()

    def _load_config_file(self, tokamakName, mode, reflections, eqFileName) -> None:
        """
        Loads the configuration file for the given tokamak
        """

        pathFileName = os.path.join(
            EMIS3D_TOKMAK_DIRECTORY, tokamakName, f"{tokamakName}_settings.yaml"
        )

        # --- Load the configuration file, if it exists
        if os.path.isfile(pathFileName):
            self.info = config_loader(pathFileName)
        else:
            print(
                f"Could not load the configuration file, file does not exist: {pathFileName}"
            )

        # --- Create the self.info dict if the file is not loaded
        if self.info == None:
            self.info = {}

        # Angle conventions used in each tokamak are different from that used in Cherab.
        # Emis3D uses the Cherab angle convention. This angle is subtracted in the evaluate
        # statements in RadDist to make the angles match.
        torConventionPhis = {"JET": np.pi / 2.0, "SPARC": 0.0, "DIII-D": 0.0}
        self.info["torConventionPhi"] = torConventionPhis[tokamakName]
        self.info["tokamakName"] = tokamakName
        self.info["mode"] = mode
        self.info["reflections"] = reflections
        self.info["eqFileName"] = eqFileName

    def _tokamak_startup(self) -> None:
        """
        This definition will:
        1. Load the equilibrium file, if given
        1. Load the wall file, defaults to what is in the equilibrium file
        3. Builds the tokamak (if mode = Build)
        4. Load the bolometers
        """

        if self.info is None:
            print("No tokamak information loaded, cannot continue!")
            return

        # --- Load the equilibrium file
        if self.info["eqFileName"] is not None:
            self._load_eqFile()

        self._load_first_wall()
        self.world = World()
        if self.info["mode"].upper() == "BUILD":
            self._build_tokamak()

    def _load_eqFile(self) -> None:
        """
        Definition loads the equilibrium file, then uses the wall information, if it is there
        """
        pathFileName = ""
        if self.info is None:
            print("No tokamak information loaded, cannot continue!")
            return

        try:
            pathFileName = os.path.join(
                EMIS3D_INPUTS_DIRECTORY,
                self.info["tokamakName"],
                "eqdsks",
                self.info["eqFileName"],
            )
            if os.path.isfile(pathFileName):
                with open(pathFileName) as f:
                    self.gfile = geqdsk.read(f)
                if self.verbose:
                    print(f"Loaded equilibrium file: {pathFileName}")
            else:
                print(f"Equilibrium file not found!")
        except Exception as e:
            print(f"Could not read the equlibrium file, error: {e}")
            print(f"Tried to read it here: {pathFileName}")

    def _load_first_wall(self) -> None:
        """
        Loads the first wall from the given text file. The file name should be within the
        tokamak setup file ["MACHINE"]["wallFileName"], and the file should be within the
        EMIS3D_TOKMAK_DIRECTORY/inputs/...txt
        """

        # --- Checkers
        if self.info is None:
            print("No tokamak information loaded, cannot continue!")
            return

        rzarray = None

        # --- Default to the one from the eqFile, if it exists
        if hasattr(self, "gfile"):
            if hasattr(self.gfile, "rlim") and hasattr(self.gfile, "zlim"):
                if self.gfile.rlim is not None and self.gfile.zlim is not None:
                    rzarray = np.vstack((self.gfile.rlim, self.gfile.zlim)).T
                else:
                    rzarray = None

        # --- Load the wall from the text file
        elif "wallFileName" in self.info["MACHINE"]:
            pathFileName = os.path.join(
                self.input_dir, self.info["MACHINE"]["wallFileName"]
            )
            try:
                rzarray = np.loadtxt(pathFileName, skiprows=0)
            except:
                rzarray = np.loadtxt(pathFileName, delimiter=",", skiprows=0)

        else:
            self.wall = None

        # --- Store the wall information
        if rzarray is not None:
            self.wall = {}
            self.wall["rzarray"] = np.array(rzarray)
            self.wall["minr"] = min(rzarray[:, 0])
            self.wall["maxr"] = max(rzarray[:, 0])
            self.wall["minz"] = min(rzarray[:, 1])
            self.wall["maxz"] = max(rzarray[:, 1])
            self.wall["wallcurve"] = path.Path(rzarray)

    def _build_tokamak(self, load_stl=False) -> None:
        """
        Builds the tokmak within the raysect world.
        """
        PFC_STL_PATH = ""

        if self.info is None:
            print("No tokamak information loaded, cannot continue!")
            return

        # Building a closed universe

        if hasattr(self, "wall") and load_stl == False:
            if self.wall is not None:
                # --- Outer wall
                Cylinder(
                    radius=self.wall["maxr"] + 0.1,
                    height=self.wall["maxz"] * 3.0,
                    material=AbsorbingSurface(),
                    name="Outer wall",
                    parent=self.world,
                    transform=translate(0, 0, (-1) * (self.wall["maxz"] * 3.0) / 2.0),
                )
                # Inner wall
                Cylinder(
                    radius=self.wall["minr"],
                    height=self.wall["maxz"] * 3.0,
                    material=AbsorbingSurface(),
                    name="Inner wall",
                    parent=self.world,
                    transform=translate(0, 0, (-1) * (self.wall["maxz"] * 3.0) / 2.0),
                )

                # Top and bottom of tokamak
                Box(
                    lower=Point3D(-10, -10, self.wall["maxz"] - 0.1),
                    upper=Point3D(10, 10, self.wall["maxz"]),
                    parent=self.world,
                    material=AbsorbingSurface(),
                    name="Top of machine",
                )
                Box(
                    lower=Point3D(-10, -10, self.wall["minz"] - 0.1),
                    upper=Point3D(10, 10, self.wall["minz"]),
                    parent=self.world,
                    material=AbsorbingSurface(),
                    name="Bottom of machine",
                )

        # --- Load the CAD file
        elif load_stl:
            try:
                # --- Standard scale if the machine is in meters
                STL_SCALE = 1.0
                if self.info["MACHINE"]["STL_UNITS"].lower() == "mm":
                    STL_SCALE = 1.0e-3

                PFC_STL_PATH = os.path.join(
                    self.input_dir, "CAD_stl_files", "d3d_CAD_full.stl"
                )
                if os.path.isfile(PFC_STL_PATH):
                    pfcs = import_stl(PFC_STL_PATH, scaling=STL_SCALE)
                    # pfcs.transform=rotate_x(90)
                    pfcs.transform = rotate_z(60)  # for r_li first wall with cutouts
                    pfcs.material = RoughTungsten(0.6)
                    pfcs.name = "PFCs"
                    pfcs.parent = self.world

                    if self.info["reflections"] == False:
                        for child in self.world.children:
                            child.material = AbsorbingSurface()
            except Exception as e:
                print(f"Could not load the CAD tokamak file: {PFC_STL_PATH}")
                print(f"Error: {e}")

        else:
            print("Building the tokamak failed")
            print(
                f"Could not load the stl file for this tokamak, {PFC_STL_PATH} or the wall"
            )
            print("was not loaded")

    def _load_bolometers(self) -> None:
        """
        Loads the bolometers for the given tokamak.

        The files should be in the tokamak input folder,
        /tokamaks/DIII-D/inputs/sxrInfo/

        """
        self.bolometers = []
        if self.info is None:
            print("No tokamak information loaded, cannot continue!")
            return

        for bolo in self.info["BOLOMETERS"]:
            b_ = Bolometer(
                world=self.world,
                tokamakName=self.info["tokamakName"],
                configFileName=self.info["BOLOMETERS"][bolo]["configFileName"],
            )
            self.bolometers.append(b_)

    def _plot_first_wall(self, ax=None) -> None:
        """
        Plots the first wall from self.wall["wallcurve"].

        Input: ax : A subplot of the plt.figure()
            f = plt.figure()
            ax = f.add_subplot(111)
        """
        if self.wall is None:
            print("No wall information loaded, cannot continue!")
            return

        show_plt = False
        if ax is None:
            f = plt.figure(figsize=(4, 7))
            ax = f.add_subplot(111)
            show_plt = True

        ax.plot(
            self.wall["wallcurve"].vertices[:, 0],
            self.wall["wallcurve"].vertices[:, 1],
            color="black",
            linewidth=2.0,
        )
        ax.set_xlabel("R [m]", fontsize=12)
        ax.set_ylabel("z [m]", fontsize=12)
        ax.set_aspect("equal")

        if show_plt:
            plt.show()

    def _plot_labels(self, ax) -> None:
        """
        Adds labels to a 3d plot, used for self.plot
        """

        if self.wall is None:
            print("No wall information loaded, cannot continue!")
            return

        # --- Generate points for the circle
        theta = np.linspace(0, 2 * np.pi, 100)
        x = self.wall["maxr"] * np.cos(theta)
        y = self.wall["maxr"] * np.sin(theta)
        z = np.zeros_like(theta)

        ax.plot(x, y, z, color="black")

        # --- Define the angles (in degrees) where labels and tick marks should be added
        angles_deg = [0, 90, 135, 180, 225, 270]
        tick_length = 0.1  # Length of each tick mark

        for angle in angles_deg:
            angle_rad = np.deg2rad(angle)  # Convert degrees to radians

            # Coordinates on the circle
            x_circle = self.wall["maxr"] * np.cos(angle_rad)
            y_circle = self.wall["maxr"] * np.sin(angle_rad)
            z_circle = 0  # Always on z = 0 plane

            # Compute the end point of the tick mark (extending outward from the circle)
            x_tick = self.wall["maxr"] * np.cos(angle_rad) * (1 + tick_length)
            y_tick = self.wall["maxr"] * np.sin(angle_rad) * (1 + tick_length)
            z_tick = 0  # Still on the z = 0 plane

            # Add the label at the circle
            ax.text(
                x_circle,
                y_circle,
                z_circle,
                f"{angle}°",
                fontsize=12,
                color="red",
                horizontalalignment="center",
                verticalalignment="center",
            )

            # Draw the tick mark as a short line segment from the circle to the tick endpoint
            ax.plot(
                [x_circle, x_tick],
                [y_circle, y_tick],
                [z_circle, z_tick],
                color="black",
                lw=2,
            )

    def _plot_bolometers(self, ax, boloName) -> None:
        """
        Plots the chords for a specific bolometer
        """

        # --- Find the correct index
        bolo_tokamak = []
        for bolo in self.bolometers:
            bolo_tokamak.append(bolo.name)

        indx_ = bolo_tokamak.index(boloName)
        for foil in self.bolometers[indx_].bolometer_camera:
            slit_centre = foil.slit.centre_point
            slit_centre_rz = point3d_to_rz(slit_centre)
            ax.plot(slit_centre_rz[0], slit_centre_rz[1], "ko")
            origin, hit, _ = foil.trace_sightline()
            centre_rz = point3d_to_rz(foil.centre_point)
            ax.plot(centre_rz[0], centre_rz[1], "kx")
            origin_rz = point3d_to_rz(origin)
            hit_rz = point3d_to_rz(hit)
            ax.plot([origin_rz[0], hit_rz[0]], [origin_rz[1], hit_rz[1]], "k")
            ax.text(
                hit_rz[0],
                hit_rz[1],
                int(foil.name[-2:]),
                fontsize="10",
                ha="center",
                va="center",
                weight="bold",
            )
            ax.set_title(boloName)

    def plot(self, fieldLineStartPhi=None) -> None:
        """
        Plot the tokamak configuration in 3D

        Inputs:
            fieldLineStartPhi :: float
                                 field line start location in degrees
        """
        if self.wall is None:
            print("No wall information loaded, cannot continue!")
            return
        if self.info is None:
            print("No tokamak information loaded, cannot continue!")
            return

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        self._plot_labels(ax)

        r = self.wall["wallcurve"].vertices[:, 0]
        z = self.wall["wallcurve"].vertices[:, 1]
        r = np.concatenate((r, [r[0]]))
        z = np.concatenate((z, [z[0]]))
        y = np.squeeze(np.zeros((1, len(r))))
        ax.plot(r, y, z, "black")
        ax.plot(r * (-1), y, z, "black")
        ax.plot(y, r, z, "black")
        ax.plot(y * (-1), r * (-1), z, "black")

        if hasattr(self, "bolometers"):

            if self.info["mode"] != "Build":
                print(
                    f"Building the tokamak! We need this to trace the chords for the bolometers"
                )
                self._build_tokamak()

            for bolo in self.bolometers:
                for foil in bolo.bolometer_camera:
                    slit_centre = foil.slit.centre_point
                    ax.plot(slit_centre[0], slit_centre[1], slit_centre[2], "ko")
                    origin, hit, _ = foil.trace_sightline()
                    ax.plot(
                        foil.centre_point[0],
                        foil.centre_point[1],
                        foil.centre_point[2],
                        "kx",
                    )
                    ax.plot(
                        [origin[0], hit[0]],
                        [origin[1], hit[1]],
                        [origin[2], hit[2]],
                        "k",
                    )
                    ax.text(hit[0], hit[1], hit[2], foil.name)

        # --- Plot the given field line
        if hasattr(self, "fieldLines"):
            colors = ["red", "green", "blue", "orange", "purple", "brown"]
            if str(fieldLineStartPhi) in self.get_fieldLines_startPhis():
                for ii, dir_ in enumerate(
                    self.fieldLines[f"{fieldLineStartPhi}"]["directionNames"]
                ):
                    line_ = self.fieldLines[f"{fieldLineStartPhi}"][dir_]
                    ax.plot(
                        line_["x"],
                        line_["y"],
                        line_["z"],
                        color=colors[ii],
                        label=dir_,
                        linewidth=2.0,
                    )

            else:
                print(f"Input fieldLinePhi of {fieldLineStartPhi}, not availble!")
                print(f"Possible fieldLinePhi(s): {self.get_fieldLines_startPhis()}")

        ax.legend()
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")  # pyright: ignore[reportAttributeAccessIssue]
        ax.set_xlim(float(-2.5), float(2.5))
        ax.set_ylim(float(-2.5), float(2.5))
        ax.set_zlim(float(-2.5), float(2.5))  # type: ignore

        plt.show()

    def set_fieldlines(
        self, startR=[], startZ=[], startPhi=0.0, numTransists=1.0
    ) -> None:
        """
        Calculate the field line progress clockwise and counterclockwise from
        StartPhi. This supports inputs of multiple R and z locations.

        Input:
        startR      : Array of R points to start the field line
        startZ      : Array of z points to start the field line
        startPhi    : Phi start location of this field line in radians

        Stores:
        self.fieldLines[startPhi] : {}
        self.fieldLines[startPhi]['R']      : 2D array of R points for each phi progression [R, phi]
        self.fieldLines[startPhi]['z']      : 2D array of z points for each phi progression [z, phi]
        self.fieldLines[startPhi]['phi']    : 1D array of phi points
        self.fieldLines[startPhi]['L']      : 1D array of the distance to this phi location [m]
        self.fieldLInes[startPhi][startR, startZ, startPhi] : stores the intial data points

        This is done so you can get an array of all of the R and z points at a particular phi location,
        this vectorized format should be faster than storing them in a different manner.
        """

        startPhideg = f"{int(np.rad2deg(startPhi))}"
        # --- Initilize the arrays
        if not hasattr(self, "fieldLines"):
            self.fieldLines = {}
        self.fieldLines[startPhideg] = {}
        self.fieldLines[startPhideg]["NumTransists"] = numTransists
        self.fieldLines[startPhideg]["directionNames"] = []
        self.fieldLines[startPhideg]["startR"] = startR
        self.fieldLines[startPhideg]["startZ"] = startZ
        self.fieldLines[startPhideg]["startPhi"] = startPhi

        # --- Loop over the direction
        direction_prefix = ["counterClock", "clockwise"]
        numTrans = [numTransists, (-1.0) * numTransists]
        for ii, direction_prefix in enumerate(direction_prefix):
            # --- Loop over the R, z coordinates, store the result
            for jj in range(len(startR)):
                if self.verbose:
                    print(
                        f"Calculating fields in the {direction_prefix[ii]} direction from\tR={startR[jj]:.2f}m, z={startZ[jj]:.2f}m"
                    )
                tracer = Fieldline_Tracer(
                    StartR=startR[jj],
                    StartZ=startZ[jj],
                    StartPhi=startPhi,
                    gfile=self.gfile,
                    NumTor=500,
                )
                # --- Trace the field line in the given direction
                tracer.trace(NumTransits=numTrans[ii])

                # --- Find the components for each revolution
                d_ = tracer.data
                rev = split_revolutions(
                    d_["x"], d_["y"], d_["z"], d_["phi"], d_["R"], d_["L"]
                )

                # --- Initilize the arrays
                for kk in range(int(numTransists)):
                    direction = f"{direction_prefix}_rev{kk}"
                    self.fieldLines[startPhideg]["directionNames"].append(direction)
                    if direction not in self.fieldLines[startPhideg]:
                        self.fieldLines[startPhideg][direction] = {}
                        for val in ["R", "L", "x", "y", "z"]:
                            self.fieldLines[startPhideg][direction][val] = np.zeros(
                                (len(startR), len(rev[kk]["phi"]))
                            )

                    # --- Store the data
                    for val in ["R", "L", "x", "y", "z"]:
                        self.fieldLines[startPhideg][direction][val][jj, :] = rev[kk][
                            val
                        ].flatten()
                    # --- Phi should be the same for each one of them, so we only need to store it once
                    if jj == 0:
                        self.fieldLines[startPhideg][direction]["phi"] = rev[kk]["phi"]

    def get_fieldLines_startPhis(self) -> list:
        return list(self.fieldLines.keys())

    def find_RZ_Fline(self, startPhi, emissionName, inputPhis=[]):
        """
        Returns the R and z arrays for the given input inputPhi location.

        Phi should always be positive!

        Method based on Divakr's answer here, this should be faster than the simple np.abs(x - x0).argmin():
        https://stackoverflow.com/questions/45349561/find-nearest-indices-for-one-array-against-all-values-in-another-array-python
        """

        B = np.array(self.fieldLines[startPhi][emissionName]["phi"])
        A = np.array(inputPhis)
        L = np.array(B).size
        sidx_B = B.argsort()
        sorted_B = B[sidx_B]
        sorted_idx = np.searchsorted(sorted_B, A)
        sorted_idx[sorted_idx == L] = L - 1
        mask = (sorted_idx > 0) & (
            (np.abs(A - sorted_B[sorted_idx - 1]) < np.abs(A - sorted_B[sorted_idx]))
        )
        flInd = sidx_B[sorted_idx - mask]
        R = self.fieldLines[startPhi][emissionName]["R"][:, flInd]
        z = self.fieldLines[startPhi][emissionName]["z"][:, flInd]
        return R, z

    def create_cameras(self, dtheta=10, dtheta_camera=4.9, phi=0):
        """
        Creates cherab cameras around the vacuum vessel.
        Currently works for DIII-D, need to test other tokamaks
        TODO:
        1. Remove R = 1.5, this is the length of the "spoke" since we use a spoke
        pattern from the center of the vessel to equally space the cameras around the vessel.
        This should just be the minor radius + some
        """

        # Plot first wall curve
        tok_r, tok_z = [0], [0]
        if self.wall is not None:
            r = self.wall["wallcurve"].vertices[:, 0]
            z = self.wall["wallcurve"].vertices[:, 1]
        r0 = (np.max(tok_r) - np.min(tok_r)) / 2.0 + np.min(tok_r)
        z0 = (np.max(tok_z) - np.min(tok_z)) / 2.0

        MACHINE_AXIS_3D = Point3D(r0, 0.0, z0)

        # --- Create the spoke pattern for the cameras, the upper and lower are the camera's width
        lines = {}
        lines_upper = {}
        lines_lower = {}

        for angle in np.arange(0, 360, dtheta):
            lines[angle] = draw_radial_lines(r0, z0, R=1.5, angle_deg=angle)

            lines_upper[angle] = draw_radial_lines(
                r0, z0, R=1.5, angle_deg=angle, initial_offset=dtheta_camera
            )
            lines_upper[angle]["offset"] = dtheta_camera
            lines_lower[angle] = draw_radial_lines(
                r0, z0, R=1.5, angle_deg=angle, initial_offset=-dtheta_camera
            )
            lines_lower[angle]["offset"] = -dtheta_camera

        # --- Fit each segement to a higher resolution, depending on the distance between the two segements
        r_ = []
        z_ = []
        for ii in range(len(tok_r)):
            if ii == len(tok_r) - 1:
                loc_ = 0
            else:
                loc_ = ii + 1
            r_.append(tok_r[ii])
            z_.append(tok_z[ii])

            # --- Length of this segment
            ds = np.sqrt(
                (tok_r[loc_] - tok_r[ii]) ** 2 + (tok_z[loc_] - tok_z[ii]) ** 2
            )

            if ds > 1.0:
                npts = 40
            else:
                npts = 20
            if ds > 0.04:
                # --- Fit the points
                if tok_r[loc_] == tok_r[ii]:
                    nx = [tok_r[loc_]] * npts
                else:
                    nx = np.linspace(tok_r[loc_], tok_r[ii], npts)
                ny = np.poly1d(
                    np.polyfit([tok_r[loc_], tok_r[ii]], [tok_z[loc_], tok_z[ii]], 1)
                )(nx)

                for x_ in nx:
                    r_.append(x_)
                for y_ in ny:
                    z_.append(y_)
        tok_r = r_.copy()
        tok_z = z_.copy()

        # --- Find where the center of camera intersects the vessel
        self.cameras = {}
        for ii, line in enumerate(lines):
            self.cameras[ii] = {}
            self.cameras[ii]["theta"] = line
            # r, z = Util_D3D.find_intersection(lines[line], tok_r, tok_z)
            # cameras[ii]["detector_center"] = Point3D(r, 0.0, z)
            r, z = find_intersection(lines_upper[line], tok_r, tok_z)
            x, y, z = rz_to_xyz(r, z, phi)
            self.cameras[ii]["p1"] = Point3D(x, y, z)

            r, z = find_intersection(lines_lower[line], tok_r, tok_z)
            x, y, z = rz_to_xyz(r, z, phi)
            self.cameras[ii]["p2"] = Point3D(x, y, z)

            self.cameras[ii]["y_vector_full"] = self.cameras[ii]["p1"].vector_to(
                self.cameras[ii]["p2"]
            )
            self.cameras[ii]["y_vector"] = self.cameras[ii]["y_vector_full"].normalise()
            self.cameras[ii]["y_width"] = self.cameras[ii]["y_vector_full"].length
            self.cameras[ii]["detector_center"] = (
                self.cameras[ii]["p1"] + self.cameras[ii]["y_vector_full"] * 0.5
            )

            x, y, z = rz_to_xyz(MACHINE_AXIS_3D[0], MACHINE_AXIS_3D[1], phi)
            self.cameras[ii]["normal_vector"] = (
                self.cameras[ii]["detector_center"].vector_to(Point3D(x, y, z))
            ).normalise()  # inward pointing
