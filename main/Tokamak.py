#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:43:28 2023

@author: bsteinlu

Updated and re-written during the refactor - JLH

TODO:
1) Currently the field lines are stored based on toroidal start location, this may be problematic for the
helical radDist when initilizing many field lines at the same location. Solution: Store the field lines
under {startPhi}_R_{startR:.2f}_z_{startz:.2f}, instead of just {startPhi}. OR... is this really an issue?
2) update self.synth_camera_test so the figure will output to a better location
3) Update create_cameras, I don't know what tok_r and tok_z are doing and they are re-assigned half-way through the definition
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from freeqdsk import geqdsk
from matplotlib import path
from raysect.core import Point3D
from raysect.core.math import rotate_z, translate
from raysect.optical import World
from raysect.optical.library import RoughTungsten
from raysect.optical import AbsorbingSurface, NullMaterial  # type: ignore

from raysect.primitive import Cylinder, import_stl, Subtract

from main.Diagnostic import Bolometer
from main.Fieldline_Tracer import Fieldline_Tracer
from main.Globals import (
    EMIS3D_TOKMAK_DIRECTORY,
    EMIS3D_INPUTS_DIRECTORY,
    SUPPORTED_TOKAMAKS,
    TOR_CONVENTION_PHIS,
)

logger = logging.getLogger(__name__)


from main.Util import (
    chord_rz_projection,
    config_loader,
    fieldline_key,
    fit_line_to_rz,
    point3d_to_rz,
    draw_radial_lines,
    find_intersection,
    rz_to_xyz,
    split_revolutions,
    compute_etendue_metric,
    get_rectangle_corners,
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
        ../tokamaks/{DIII-D, JET, SPARC, etc}/{DIII-D, JET, SPARC, etc}_settings.yaml

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
            logger.debug(f"Please enter a valid tokamak name!")
            logger.debug(f"Tokamaks currently supported are: {SUPPORTED_TOKAMAKS}")
            raise Exception

        else:
            # --- Load the configuration file
            self._load_config_file(tokamakName, mode, reflections, eqFileName)

            # --- Set the general input directory
            self.input_dir = EMIS3D_TOKMAK_DIRECTORY / tokamakName / "inputs"

            # --- Run the startup program
            self._tokamak_startup(loadBolometers=loadBolometers)

    def _load_config_file(self, tokamakName, mode, reflections, eqFileName) -> None:
        """
        Loads the configuration file for the given tokamak
        """

        pathFileName = (
            EMIS3D_TOKMAK_DIRECTORY / tokamakName / f"{tokamakName}_settings.yaml"
        )

        # --- Load the configuration file, if it exists
        self.info = None
        if os.path.isfile(pathFileName):
            self.info = config_loader(pathFileName)
        else:
            logger.warning(
                f"Could not load the configuration file, file does not exist: {pathFileName}"
            )

        # --- Create the self.info dict if the file is not loaded
        if self.info is None:
            self.info = {}

        try:
            torConvention = TOR_CONVENTION_PHIS.get(tokamakName)
        except Exception:
            raise RuntimeError(
                f"!!! Error, put in '{tokamakName}' in TOR_CONVENTION_PHIS in the globals.py file!"
            )

        self.info["torConventionPhi"] = torConvention
        self.info["tokamakName"] = tokamakName
        self.info["mode"] = mode
        self.info["reflections"] = reflections
        self.info["eqFileName"] = eqFileName

    def _tokamak_startup(self, loadBolometers=False) -> None:
        """
        This definition will:
        1. Load the equilibrium file, if given
        2. Load the bolometers
        3. Load the wall file, defaults to what is in the equilibrium file
        4. Builds the tokamak (if mode = Build)
        """

        if self.info is None:
            logger.error("No tokamak information loaded, cannot continue!")
            return

        # --- Create the raysect world
        self.world = World()

        # --- Load the equilibrium file
        if self.info["eqFileName"] is not None:
            self._load_eqFile()

        # --- Load the bolometers
        if loadBolometers:
            self._load_bolometers()

        # --- Load the first wall
        self._load_first_wall()

        # --- Create boundries in the raysect world
        if self.info["mode"].upper() == "BUILD":
            self._build_tokamak()

    def _load_eqFile(self) -> None:
        """
        Definition loads the equilibrium file, then uses the wall information, if it is there
        """
        pathFileName = ""
        if self.info is None:
            logger.error("No tokamak information loaded, cannot continue!")
            return

        try:
            pathFileName = (
                EMIS3D_INPUTS_DIRECTORY
                / self.info["tokamakName"]
                / "eqdsks"
                / self.info["eqFileName"]
            )

            if os.path.isfile(pathFileName):
                with open(pathFileName) as f:
                    self.gfile = geqdsk.read(f)
                if self.verbose:
                    logger.debug(f"Loaded equilibrium file: {pathFileName}")
            else:
                logger.debug(f"Equilibrium file not found!")
        except Exception as e:
            logger.error(f"Could not read the equilibrium file, error: {e}")
            logger.debug(f"Tried to read it here: {pathFileName}")

    def _load_first_wall(self) -> None:
        """
        Loads the first wall from the given text file. The file name should be within the
        tokamak setup file ["MACHINE"]["wallFileName"], and the file should be within the
        EMIS3D_TOKMAK_DIRECTORY/inputs/...txt
        """

        # --- Checkers
        if self.info is None:
            logger.error("No tokamak information loaded, cannot continue!")
            return

        rzarray = None

        # --- Default to the one from the eqFile, if it exists
        if hasattr(self, "gfile"):
            if hasattr(self.gfile, "rlim") and hasattr(self.gfile, "zlim"):
                if self.gfile.rlim is not None and self.gfile.zlim is not None:
                    rzarray = np.vstack((self.gfile.rlim, self.gfile.zlim)).T
                else:
                    rzarray = None

        # --- Fall back to the wall text file if the eqFile had no limiter data
        if rzarray is None and "wallFileName" in self.info["MACHINE"]:
            pathFileName = self.input_dir / self.info["MACHINE"]["wallFileName"]

            try:
                rzarray = np.loadtxt(pathFileName, skiprows=0)
            except ValueError:
                rzarray = np.loadtxt(pathFileName, delimiter=",", skiprows=0)

        # --- Store the wall information
        if rzarray is not None:
            self.wall = {}
            self.wall["rzarray"] = np.array(rzarray)
            self.wall["minr"] = min(rzarray[:, 0])
            self.wall["maxr"] = max(rzarray[:, 0])
            self.wall["minz"] = min(rzarray[:, 1])
            self.wall["maxz"] = max(rzarray[:, 1])
            self.wall["wallcurve"] = path.Path(rzarray)
        else:
            self.wall = None

    def _build_tokamak(self, load_stl=False) -> None:
        """
        Builds the tokmak within the raysect world.
        """
        PFC_STL_PATH = ""

        if self.info is None:
            logger.error("No tokamak information loaded, cannot continue!")
            return

        self.cherab_objects = {}

        # Building a closed universe
        if hasattr(self, "wall") and load_stl == False:
            if self.wall is not None:
                # --- Find the z limits
                if "Bolometer Z Limits" in self.info:
                    minz = np.min(
                        [self.wall["minz"], self.info["Bolometer Z Limits"][0]]
                    ).astype(float)
                    maxz = np.max(
                        [self.wall["maxz"], self.info["Bolometer Z Limits"][1]]
                    ).astype(float)
                else:
                    minz = self.wall["minz"]
                    maxz = self.wall["maxz"]
                height = np.abs(maxz) + np.abs(minz)
                offset = minz

                # --- Find the R limits
                if "Bolometer R Limits" in self.info:
                    minR = np.min(
                        [self.wall["minr"], self.info["Bolometer R Limits"][0]]
                    ).astype(float)
                    maxR = np.max(
                        [self.wall["maxr"], self.info["Bolometer R Limits"][1]]
                    ).astype(float)
                else:
                    minR = self.wall["minr"]
                    maxR = self.wall["maxr"]

                # --- Outer wall
                cylinder_outer = Cylinder(
                    radius=maxR + 0.2,
                    height=height + 0.2,
                    name="Outer wall",
                )

                # --- Inner wall
                # NOTE: This should be modified if any bolometers are inside of this radius,
                # otherwise Raysect will not trace the chords correctly
                cylinder_inner = Cylinder(
                    radius=minR - 0.2,
                    height=height + 0.2,
                    name="Inner wall",
                )

                self.cherab_objects["Tokamak Wall"] = Subtract(
                    cylinder_outer,
                    cylinder_inner,
                    material=AbsorbingSurface(),  # Do NOT CHANGE THIS to a NullSurface, otherwise the foil.observe will not work properly
                    name="Tokamak Wall",
                    parent=self.world,
                    transform=translate(0, 0, offset - 0.1),
                )

                # --- Have the emission surface inside the tokamak
                # --- Outer wall
                emiss_outer = Cylinder(
                    radius=self.wall["maxr"] + 1.0,
                    height=height,
                    name="Outer emission wall",
                )

                # --- Inner wall
                emiss_inner = Cylinder(
                    radius=self.wall["minr"],
                    height=height,
                    name="Inner emission wall",
                )

                self.cherab_objects["Emission Surface"] = Subtract(
                    emiss_outer,
                    emiss_inner,
                    name="Emission Surface",
                    parent=self.world,
                    transform=translate(0, 0, offset),
                    material=NullMaterial(),
                )

        # --- Load the CAD file
        elif load_stl:
            try:
                # --- Standard scale if the machine is in meters
                STL_SCALE = 1.0
                if self.info["MACHINE"]["STL_UNITS"].lower() == "mm":
                    STL_SCALE = 1.0e-3

                PFC_STL_PATH = (
                    self.input_dir
                    / "CAD_stl_files"
                    / self.info["MACHINE"]["PFC_STL_PATH"]
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
                logger.warning(f"Could not load the CAD tokamak file: {PFC_STL_PATH}")
                logger.error(f"Error: {e}")

        else:
            logger.error("Building the tokamak failed")
            logger.warning(
                f"Could not load the stl file for this tokamak, {PFC_STL_PATH} or the wall"
            )
            logger.debug("was not loaded")

    def _load_bolometers(self) -> None:
        """
        Loads the bolometers for the given tokamak.

        The files should be in the tokamak input folder,
        /tokamaks/DIII-D/inputs/sxrInfo/

        """
        self.bolometers = []

        if self.info is None:
            logger.error("No tokamak information loaded, cannot continue!")
            return
        # --- Store all of the bolometer groups, used to make plotting easier
        self.info["Bolometer Groups"] = []

        # --- Find the maximum and minimum locations of the bolometer, used to create the world
        r_limits = [1000, 0]
        z_limits = [1000, -1000]

        # --- Create each bolometer in the tokamak configuration file
        for bolo in self.info["BOLOMETERS"]:
            b_ = Bolometer(
                world=self.world,
                tokamakName=self.info["tokamakName"],
                configFileName=self.info["BOLOMETERS"][bolo]["configFileName"],
            )
            self.bolometers.append(b_)

            if b_.info is not None:
                offset = np.abs(b_.info["SLIT_SENSOR_SEPARATION"])
                if b_.info["CAMERA_POSITION_R_Z_PHI"][0] + offset > r_limits[1]:
                    r_limits[1] = b_.info["CAMERA_POSITION_R_Z_PHI"][0] + offset
                if b_.info["CAMERA_POSITION_R_Z_PHI"][0] - offset < r_limits[0]:
                    r_limits[0] = b_.info["CAMERA_POSITION_R_Z_PHI"][0] - offset
                if b_.info["CAMERA_POSITION_R_Z_PHI"][1] + offset > z_limits[1]:
                    z_limits[1] = b_.info["CAMERA_POSITION_R_Z_PHI"][1] + offset
                if b_.info["CAMERA_POSITION_R_Z_PHI"][1] - offset < z_limits[0]:
                    z_limits[0] = b_.info["CAMERA_POSITION_R_Z_PHI"][1] - offset

                if b_.info["GROUP_NAME"] not in self.info["Bolometer Groups"]:
                    self.info["Bolometer Groups"].append(b_.info["GROUP_NAME"])

        # --- Used to create the cherab bounding box for the bolometers
        self.info["Bolometer R Limits"] = r_limits
        self.info["Bolometer Z Limits"] = z_limits

    def _inside_tokamak(self, points: np.ndarray) -> np.ndarray:
        """
        Check whether (R, z) points lie within the tokamak wall.

        Parameters
        ----------
        points : np.ndarray, shape (N, 2) — column-stacked [R, z] coordinates.

        Returns
        -------
        inside : np.ndarray of bool, shape (N,)
        """
        if self.wall is None:
            raise RuntimeError(
                "Tokamak wall not initialised — ensure _load_first_wall() "
                "has been called before using _inside_tokamak()."
            )
        return self.wall["wallcurve"].contains_points(points)

    def _make_raysect_surface_transparent(self, surfaceName="") -> None:
        """
        Makes the raysect object transparent. This is done after observation in order
        for the sightlines to trace properly
        """
        # --- Add the emitter to the tokamak wall
        for val in self.world.children:
            if val.name == surfaceName:
                val.material = NullMaterial()

    def _change_World_surface_material(self, material, name="Emission Surface") -> None:
        """
        Changes the material of the emission surface, used for testing purposes
        """
        if self.info is None:
            logger.error("No tokamak information loaded, cannot continue!")
            return

        if name not in ["Emission Surface", "Tokamak Wall"]:
            logger.debug(
                "Name must be one of these two: Emission Surface, Tokamak Wall"
            )
            return

        for child in self.world.children:
            if child.name == name:
                child.material = material

    def _plot_first_wall(self, ax=None) -> None:
        """
        Plots the first wall from self.wall["wallcurve"].

        Input: ax : A subplot of the plt.figure()
            f = plt.figure()
            ax = f.add_subplot(111)
        """
        if self.wall is None:
            logger.error("No wall information loaded, cannot continue!")
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
            logger.error("No wall information loaded, cannot continue!")
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

    def _plot_channel_with_envelope(
        self, ax, foil, slit, length=None, debug=False
    ) -> None:

        if ax is not None:

            if debug:
                logger.debug("\n--- FOIL GEOMETRY ---")
                logger.debug(f"Width  : {foil.y_width:.6f} m")
                logger.debug(f"Height : {foil.x_width:.6f} m")

                logger.debug("\n--- SLIT GEOMETRY ---")
                logger.debug(f"Width  : {slit.dy:.6f} m")
                logger.debug(f"Height : {slit.dx:.6f} m")

                foil_center = foil.to_root() * foil.centre_point
                slit_center = slit.to_root() * slit.centre_point

                separation = np.sqrt(
                    (foil_center.x - slit_center.x) ** 2
                    + (foil_center.y - slit_center.y) ** 2
                    + (foil_center.z - slit_center.z) ** 2
                )

                logger.debug("\n--- FOIL to SLIT SEPARATION ---")
                logger.debug(f"Center-to-center distance: {separation:.6f} m")

            # Get corners
            foil_corners = get_rectangle_corners(foil)
            slit_corners = get_rectangle_corners(slit)

            # Identify upper/lower in Z
            foil_sorted = sorted(foil_corners, key=lambda p: p.z)
            slit_sorted = sorted(slit_corners, key=lambda p: p.z)

            foil_lower = foil_sorted[0]
            foil_upper = foil_sorted[-1]
            slit_lower = slit_sorted[0]
            slit_upper = slit_sorted[-1]

            # Two diagonal options
            option1 = [(foil_lower, slit_upper), (foil_upper, slit_lower)]

            option2 = [(foil_lower, slit_lower), (foil_upper, slit_upper)]

            # Compute metric
            metric1 = sum(compute_etendue_metric(a, b) for a, b in option1)
            metric2 = sum(compute_etendue_metric(a, b) for a, b in option2)

            if metric1 >= metric2:
                chosen = option1
            else:
                chosen = option2

            # Plot foil & slit corners
            for p in foil_corners:
                r, z = point3d_to_rz(p)
                ax.scatter(r, z)

            for p in slit_corners:
                r, z = point3d_to_rz(p)
                ax.scatter(r, z)

            # Plot envelope lines
            for a, b in chosen:
                start = np.array([a.x, a.y, a.z])
                end = np.array([b.x, b.y, b.z])

                direction = end - start
                distance = np.linalg.norm(direction)
                direction /= distance

                if length is not None:
                    new_end = start + direction * length
                else:
                    new_end = end

                r1, z1 = point3d_to_rz(a)

                temp_point = a.__class__(*new_end)
                r2, z2 = point3d_to_rz(temp_point)

                plt.plot([r1, r2], [z1, z2], linewidth=2, color="tab:red")

    @staticmethod
    def _chord_phi_values(
        value, default: float, num_chords: int, boloName: str, key: str
    ):
        """
        Normalise an optional per-chord toroidal angle entry into a list of
        num_chords floats, in degrees.

        value :: what was found in bolo.info, one of
                 None                -> every chord sits at `default`
                 a scalar            -> every chord uses that angle
                 a list of num_chords-> used as-is
        default :: fallback angle, normally the camera's own phi
        """

        if value is None:
            return [float(default)] * num_chords

        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != num_chords:
                logger.error(
                    f"{boloName}: '{key}' has {len(value)} entries but there are "
                    f"{num_chords} chords. Falling back to the camera phi of {default}."
                )
                return [float(default)] * num_chords
            return [float(v) for v in value]

        # --- A single angle shared by every chord
        return [float(value)] * num_chords

    def _plot_bolometers(
        self,
        ax,
        boloGroupName,
        plot_chord_info=False,
        plot_etendue=[],
        legend=False,
    ) -> None:
        """
        Plots the chords for a specific bolometer group
        ax :: matplotlib.plot
        boloGroupName :: The GROUP_NAME in each bolometer file
        plot_chord_info :: Plot r0, rf, etc. in each bolometer file, typically used for initial
                        debugging of new bolometers since it compares Cherab to known chord positions
        """

        # --- Change the inner wall to an absorbing surface, so the chords have something intersect with
        self._change_World_surface_material(AbsorbingSurface(), name="Tokamak Wall")

        # --- Make sure self.info is initiated
        if self.info is None:
            logger.error("No tokamak information loaded, cannot continue!")
            return

        # --- Check to see if boloGroupName is in self.info['Bolometer Groups']
        if boloGroupName not in self.info["Bolometer Groups"]:
            logger.debug(
                f"{boloGroupName} is not found in self.info['Bolometer Groups']"
            )
            return

        # --- Remove each bolometer from the world
        for bolo in self.bolometers:
            bolo._change_parent(value=None)

        # --- Loop over each bolometer, only plot those with the correct group name
        label_cherab = True
        for bolo in self.bolometers:
            bolo._change_parent(value=self.world)
            if bolo.group_name == boloGroupName:
                # --- Over plot the chords in the cofig file
                if plot_chord_info and bolo.info is not None:
                    if "r0" in bolo.info:
                        label_ = "Chords from config file"

                        num_chords = len(bolo.info["r0"])

                        # --- phi0 and phif are optional. Without them the chord
                        # --- is assumed to lie in the camera's own poloidal plane.
                        camera_phi = float(bolo.info["CAMERA_POSITION_R_Z_PHI"][2])
                        phi0_vals = self._chord_phi_values(
                            bolo.info.get("phi0"),
                            default=camera_phi,
                            num_chords=num_chords,
                            boloName=bolo.info["NAME"],
                            key="phi0",
                        )
                        phif_vals = self._chord_phi_values(
                            bolo.info.get("phif"),
                            default=camera_phi,
                            num_chords=num_chords,
                            boloName=bolo.info["NAME"],
                            key="phif",
                        )

                        for ii in range(num_chords):
                            # --- Check to see if the currnt plot already has the correct label
                            _, labels = ax.get_legend_handles_labels()
                            if "Chords from config file" in labels:
                                label_ = "__no_legend__"

                            # --- Project the 3D chord into the plane of the
                            # --- bolometer, then fit a straight line to it. For an
                            # --- in-plane chord (phi0 == phif) the projection is
                            # --- already straight and the fit returns the original
                            # --- end points.
                            R_proj, z_proj = chord_rz_projection(
                                r0=bolo.info["r0"][ii],
                                z0=bolo.info["z0"][ii],
                                phi0=phi0_vals[ii],
                                rf=bolo.info["rf"][ii],
                                zf=bolo.info["zf"][ii],
                                phif=phif_vals[ii],
                            )
                            start, end, residual = fit_line_to_rz(R_proj, z_proj)

                            if residual > 1.0e-3:
                                logger.debug(
                                    f"{bolo.info['NAME']} chord {ii}: projected path "
                                    f"bends {residual*100.0:.2f} cm away from the fitted "
                                    f"straight chord (phi0 = {phi0_vals[ii]}, "
                                    f"phif = {phif_vals[ii]})."
                                )

                            projection_label = "Projected chord path from config"
                            _, labels = ax.get_legend_handles_labels()
                            if projection_label in labels:
                                projection_label = "__no_legend__"

                            ax.plot(
                                R_proj,
                                z_proj,
                                linestyle="solid",
                                linewidth=2.0,
                                color="darkgreen",
                                label=projection_label,
                            )

                for foil in bolo.bolometer_camera.foil_detectors:

                    label = "__no_legend__"
                    if label_cherab:
                        label = "Chords from Raysect"
                        label_cherab = False

                    # --- Slit center
                    slit_rz = point3d_to_rz(foil.slit.centre_point)
                    ax.plot(slit_rz[0], slit_rz[1], "ko")

                    # --- Foil center
                    centre_rz = point3d_to_rz(foil.centre_point)
                    ax.plot(centre_rz[0], centre_rz[1], "kx")

                    # --- Ray-traced sightline
                    origin, hit, _ = foil.trace_sightline()
                    origin_rz = point3d_to_rz(origin)

                    if origin is not None and hit is not None:
                        origin_rz = point3d_to_rz(origin)
                        hit_rz = point3d_to_rz(hit)

                        ax.plot(
                            [origin_rz[0], hit_rz[0]],
                            [origin_rz[1], hit_rz[1]],
                            color="tab:blue",
                            linewidth=1.0,
                            label=label,
                        )

                        # --- Add the channel number
                        ch = ""
                        try:
                            if int(foil.name[-2:]):
                                ch = int(foil.name[-2:])
                        # For JET foils
                        except Exception:
                            n = foil.name.split("_")[1]
                            ch = n[2:]

                        ax.text(
                            hit_rz[0],
                            hit_rz[1],
                            ch,
                            fontsize="10",
                            ha="center",
                            va="center",
                            weight="bold",
                        )

                        # --- Highlight the etendue of each channel
                        if foil.name in plot_etendue:
                            dR = np.abs(origin_rz[0] - hit_rz[0])
                            dz = np.abs(origin_rz[1] - hit_rz[1])
                            length = np.sqrt(dR**2 + dz**2)

                            self._plot_channel_with_envelope(
                                ax, foil=foil, slit=foil.slit, length=length
                            )

                    # Debug arrow in ray direction
                    slit_rz = point3d_to_rz(foil.slit.centre_point)
                    direction = np.array(
                        [slit_rz[0] - origin_rz[0], slit_rz[1] - origin_rz[1]]
                    )
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                        scale = 0.05
                        ax.quiver(
                            origin_rz[0],
                            origin_rz[1],
                            direction[0] * scale,
                            direction[1] * scale,
                            angles="xy",
                            scale_units="xy",
                            scale=1,
                            color="red",
                        )
            bolo._change_parent(value=None)

        # --- Add back each bolometer from the world
        for bolo in self.bolometers:
            bolo._change_parent(value=self.world)

        if legend:
            ax.legend(loc="upper right")
        ax.set_title(boloGroupName)

    def get_ave_bolometer_tor_loc(self, boloGroupName):
        """
        Returns the average toroidal angle of the bolometer group
        """

        phi = []

        # --- Make sure self.info is initiated
        if self.info is None:
            logger.error("No tokamak information loaded, cannot continue!")
            return None

        # --- Check to see if boloGroupName is in self.info['Bolometer Groups']
        if boloGroupName not in self.info["Bolometer Groups"]:
            logger.debug(
                f"{boloGroupName} is not found in self.info['Bolometer Groups']"
            )
            return None

        # --- Loop over each bolometer, only plot add with the correct group name
        for bolo in self.bolometers:
            if bolo.info["GROUP_NAME"] == boloGroupName:
                phi.append(bolo.info["CAMERA_POSITION_R_Z_PHI"][2])

        return np.mean(np.array(phi))

    def plot(self, fieldLineStartPhi=None) -> None:
        """
        Plot the tokamak configuration in 3D

        Inputs:
            fieldLineStartPhi :: float
                                 field line start location in degrees
        """
        if self.wall is None:
            logger.error("No wall information loaded, cannot continue!")
            return
        if self.info is None:
            logger.error("No tokamak information loaded, cannot continue!")
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
                logger.debug(
                    f"→ Building the tokamak! We need this to trace the chords for the bolometers"
                )
                self._build_tokamak()
            try:
                # --- Remove each bolometer from the world
                for bolo in self.bolometers:
                    bolo._change_parent(value=None)

                for bolo in self.bolometers:
                    bolo._change_parent(value=self.world)
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
                    bolo._change_parent(value=None)

                # --- Add back each bolometer to the world
                for bolo in self.bolometers:
                    bolo._change_parent(value=self.world)
            except Exception as e:
                logger.error(f"Could not plot the bolometer chords, error: {e}")
                logger.debug(
                    f"Ensure that the tokamak is built before plotting the bolometers, currently in mode: {self.info['mode']}"
                )

        # --- Plot the given field line
        if hasattr(self, "fieldLines"):
            colors = ["red", "green", "blue", "orange", "purple", "brown"]
            fieldlineKey = fieldline_key(fieldLineStartPhi)

            if str(fieldlineKey) in self.get_fieldLines_startPhis():
                for ii, dir_ in enumerate(
                    self.fieldLines[f"{fieldlineKey}"]["directionNames"]
                ):
                    line_ = self.fieldLines[f"{fieldlineKey}"][dir_]
                    ax.plot(
                        line_["x"],
                        line_["y"],
                        line_["z"],
                        color=colors[ii],
                        label=dir_,
                        linewidth=2.0,
                    )

            else:
                logger.debug(f"Input fieldLinePhi of {fieldlineKey}, not availble!")
                logger.debug(
                    f"Possible fieldLinePhi(s): {self.get_fieldLines_startPhis()}"
                )

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

        if numTransists > 1:
            raise ValueError(
                f"ERROR! Code currently does not support more than 1 toroidal transit of the field lines!"
            )

        startPhideg = fieldline_key(startPhi, degrees=False)

        # --- Initialize the arrays
        if not hasattr(self, "fieldLines"):
            self.fieldLines = {}
        self.fieldLines[startPhideg] = {}
        self.fieldLines[startPhideg]["NumTransists"] = numTransists
        self.fieldLines[startPhideg]["directionNames"] = []
        self.fieldLines[startPhideg]["startR"] = startR
        self.fieldLines[startPhideg]["startZ"] = startZ
        self.fieldLines[startPhideg]["startPhi"] = startPhi

        # --- Loop over the direction
        direction_prefix_ = ["counterClock", "clockwise"]
        numTrans = [numTransists, (-1.0) * numTransists]
        for ii, direction_prefix in enumerate(direction_prefix_):
            # --- Loop over the R, z coordinates, store the result
            for jj in range(len(startR)):
                if self.verbose:
                    logger.debug(
                        f"Calculating fields in the {direction_prefix} direction from\tR={startR[jj]:.2f}m, z={startZ[jj]:.2f}m"
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

                # --- Initialize the arrays
                for kk in range(int(numTransists)):
                    direction = f"{direction_prefix}_rev{kk}"

                    if direction not in self.fieldLines[startPhideg]:
                        self.fieldLines[startPhideg]["directionNames"].append(direction)
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
                        phi_arr = rev[kk]["phi"]
                        self.fieldLines[startPhideg][direction]["phi"] = phi_arr
                        # Pre-compute and cache the sort index so find_RZ_Fline
                        # doesn't recompute it on every emissivity evaluation call.
                        sidx = phi_arr.argsort()
                        self.fieldLines[startPhideg][direction]["phi_sidx"] = sidx
                        self.fieldLines[startPhideg][direction]["phi_sorted"] = phi_arr[
                            sidx
                        ]

    def get_fieldLines_startPhis(self) -> list:
        return list(self.fieldLines.keys())

    def find_RZ_Fline(self, startPhi, emissionName, inputPhis=[]):
        """
        Returns the R and z arrays for the given input inputPhi location.

        Phi should always be positive!

        Method based on Divakr's answer here, this should be faster than the simple np.abs(x - x0).argmin():
        https://stackoverflow.com/questions/45349561/find-nearest-indices-for-one-array-against-all-values-in-another-array-python

        The sort index is pre-computed in set_fieldlines() and cached, so this
        method only pays the searchsorted cost on each call.
        """

        key = fieldline_key(startPhi)
        if key not in self.fieldLines:
            raise KeyError(
                f"No field lines traced at {startPhi} deg (key '{key}')"
                f"Available: {self.get_fieldLines_startPhis()}"
            )

        fline = self.fieldLines[key][emissionName]

        sidx_B = fline["phi_sidx"]
        sorted_B = fline["phi_sorted"]
        A = np.array(inputPhis)
        L = sorted_B.size
        sorted_idx = np.searchsorted(sorted_B, A)
        sorted_idx[sorted_idx == L] = L - 1
        mask = (sorted_idx > 0) & (
            (np.abs(A - sorted_B[sorted_idx - 1]) < np.abs(A - sorted_B[sorted_idx]))
        )
        flInd = sidx_B[sorted_idx - mask]
        R = fline["R"][:, flInd]
        z = fline["z"][:, flInd]
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
            lines[angle] = draw_radial_lines(r0, z0, R=1.5, angle_deg=float(angle))

            lines_upper[angle] = draw_radial_lines(
                r0, z0, R=1.5, angle_deg=float(angle), initial_offset=dtheta_camera
            )
            lines_upper[angle]["offset"] = dtheta_camera
            lines_lower[angle] = draw_radial_lines(
                r0, z0, R=1.5, angle_deg=float(angle), initial_offset=-dtheta_camera
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
            r, z = find_intersection(lines_upper[line], tok_r, tok_z)  # type: ignore
            x, y, z = rz_to_xyz(r, z, phi)  # type: ignore
            self.cameras[ii]["p1"] = Point3D(x, y, z)

            r, z = find_intersection(lines_lower[line], tok_r, tok_z)  # type: ignore
            x, y, z = rz_to_xyz(r, z, phi)  # type: ignore
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

    def _change_object_parent(self, parent=None):
        """
        When calculating the Etendue's for each bolometer,
        the parent for each other object should be set to None
        """
        for val in self.cherab_objects:
            self.cherab_objects[val].parent = parent

    def calc_etendues(self) -> None:
        """
        Sets the wall material to NullMaterial prior to calling calc_etendues for each bolometer
        """

        # --- We need to remove each object that is not the specific bolometer from the scene
        self._change_object_parent(parent=None)

        # --- Remove each bolometer from the world
        for bolo in self.bolometers:
            bolo._change_parent(value=None)

        # --- Calculate etendue for each bolometer
        for bolo in self.bolometers:
            bolo._change_parent(value=self.world)
            bolo._calc_etendues()
            bolo._change_parent(value=None)

        # --- Add each bolometer back to the world
        for bolo in self.bolometers:
            bolo._change_parent(value=self.world)

        # --- Unbuild the tokamak
        self._change_object_parent(parent=self.world)

    def synth_camera_test(
        self,
        CameraType=0,
        TorAngleDeg=0.0,
        Title="render",
        SpecBins=25,
        PixelSamples=250,
        WithWallCAD=True,
        IndLightSize=0.001,
        BoundingCylMult=1.0,
        ZoomMult=1.0,
        Zadjust=0.0,
        Radjust=0.0,
        major_radius=1.67,
        minor_radius=1.25,
    ):

        # OutputFile = os.path.join("outputs", (str(Title) + ".png"))
        OutputFile = str(Title) + ".png"

        from raysect.core.math import translate
        from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D  # type: ignore
        from raysect.optical import rotate, d65_white  # type: ignore
        from raysect.optical.library import Aluminium
        from raysect.optical.material import UniformSurfaceEmitter  # type: ignore
        from raysect.optical.library.spectra.colours import red, green, blue, purple
        from raysect.primitive import Sphere

        # add general toroidal light source
        height = 0.2
        for vertdisp in [-1.0, -0.5, 0.5, 1.0]:
            outer_cyl = Cylinder(
                major_radius * 1.02,
                height,
                transform=translate(0, 0, (-0.5 * height) + vertdisp),
            )
            inner_cyl = Cylinder(
                major_radius * 0.98,
                height * 1.2,
                transform=translate(0, 0, (-0.6 * height) + vertdisp),
            )
            illumination = Subtract(outer_cyl, inner_cyl)
            illumination.material = UniformSurfaceEmitter(d65_white, 3.0)
            illumination.parent = self.world

        if WithWallCAD:
            # load CAD from inputs directory
            logger.debug(
                "This function is only set up for DIII-D so far, it's loading the DIII-D CAD"
            )
            pfcs = import_stl(self.input_dir / "CAD_stl_files" / "d3d_CAD_full.stl")

            pfcs.material = RoughTungsten(0.6)
            pfcs.name = "PFCs"
            pfcs.parent = self.world
        else:
            # Add general bounding cylinder if not CAD
            bounding_height = minor_radius * 2.0 * 2.1
            bounding_cyl = Cylinder(
                (major_radius + (BoundingCylMult * minor_radius)) * 1.1,
                bounding_height,
                transform=translate(0, 0, -0.5 * bounding_height),
            )
            bounding_cyl.material = RoughTungsten(0.6)
            bounding_cyl.parent = self.world

            # near wall light strips:
            for vertdisp in [-1.0, -0.5, 00, 0.5, 1.0]:
                outer_cyl = Cylinder(
                    (major_radius + (BoundingCylMult * minor_radius)) * 1.08,
                    height,
                    transform=translate(0, 0, (-0.5 * height) + vertdisp),
                )
                inner_cyl = Cylinder(
                    (major_radius + (BoundingCylMult * minor_radius)) * 1.06,
                    height * 1.2,
                    transform=translate(0, 0, (-0.6 * height) + vertdisp),
                )
                illumination = Subtract(outer_cyl, inner_cyl)
                illumination.material = UniformSurfaceEmitter(d65_white, 3.0)
                illumination.parent = self.world

        rgb = RGBPipeline2D(name="sRGB", display_progress=False)
        sampler = RGBAdaptiveSampler2D(
            rgb, ratio=100, fraction=0.2, min_samples=500, cutoff=0.05
        )
        fov = 140

        # Add indicator lights and set casing material
        for bolo in self.bolometers:
            # if bolo.name == "KB5H_12":
            if True:
                bolo.bolometer_camera.camera_geometry.material = RoughTungsten(0.6)

                foils = bolo.bolometer_camera.foil_detectors
                for foil in foils:
                    center_point = foil.centre_point
                    CenterPoint = Sphere(
                        IndLightSize,
                        parent=self.world,
                        transform=translate(
                            center_point.x, center_point.y, center_point.z
                        ),
                    )
                    CenterPoint.material = UniformSurfaceEmitter(red, 100.0)
            else:
                bolo.bolometer_camera.parent = None

        if CameraType == 0:
            # Camera that looks in the counterclockwise toroidal direction
            transform = (
                rotate(0.0, 0.0, TorAngleDeg - 90.0 - (1.0 * ZoomMult))
                * translate(0.0, major_radius + Radjust, Zadjust)
                * rotate(0.0, -90.0, 0.0)
                * rotate(90.0, 0.0, 0.0)
            )
        elif CameraType == 1:
            # Camera for viewing outboard side
            transform = (
                rotate(0.0, 0.0, TorAngleDeg - 180.0)
                * translate(-1.0 * ZoomMult * major_radius, 0.0, Zadjust)
                * rotate(0.0, -90.0, 0.0)
                * rotate(90.0, 0.0, 0.0)
            )
        else:
            raise Exception("Camera type does not match an option")

        camera = PinholeCamera(
            (350, 250),
            fov=fov,
            parent=self.world,
            transform=transform,
            pipelines=[rgb],
            frame_sampler=sampler,
        )

        camera.spectral_bins = SpecBins
        camera.pixel_samples = PixelSamples

        plt.ion()
        p = 1
        while not camera.render_complete:
            logger.debug("Rendering pass {}...".format(p))
            camera.observe()
            logger.debug("")
            p += 1
            stopRender = input("Stop? (y to stop)")
            if stopRender == "y":
                break

        plt.ioff()
        # rgb.display()
        rgb.save(OutputFile)

        # --- Add each bolometer back to the world
        for bolo in self.bolometers:
            bolo._change_parent(value=self.world)
