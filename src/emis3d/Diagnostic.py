#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:43:28 2023

@author: bsteinlu

Re-wrote this during the refactor to only include a bolometer class. All of the
physical information is now stored in a self.info dictionary instead of doing
self.ax0, etc. for each variable. -JLH August, 2025.

TODO
1. Add more camera input options (from stl files, for example)
2. Remove self.load_kb5_camera eventually, this was just added to compare building the JET bolometer from primitives
"""

import os

import numpy as np
from cherab.tools.observers import BolometerCamera, BolometerFoil, BolometerSlit
from raysect.core import (
    AffineMatrix3D,
    Node,
    Point3D,
    Vector3D,
    translate,
)
from raysect.optical import AbsorbingSurface, NullMaterial  # type: ignore
from emis3d.workspace import Workspace, get_workspace
from emis3d.Util import RPhi_To_XY, config_loader, rotate_vector
from raysect.primitive import Box, Subtract


class Bolometer(object):
    """
    Basic bolometer class loads the configuration file and creates the bolometer.
    The bolometer configuration file names should be defined in the tokamak
    settings file, and each file should be in <machine>/inputs/sxrInfo/ on the
    workspace machine search path (see emis3d.workspace). A camera file in the
    user's machine folder overrides a built-in file with the same name.
    """

    def __init__(
        self,
        world=None,
        tokamakName=None,
        configFileName=None,
        workspace: Workspace | None = None,
    ):

        self.world = world
        self.configFileName = configFileName
        self.tokamakName = tokamakName
        self.ws = workspace if workspace is not None else get_workspace()
        self.config_path = None
        self._load_config_file(tokamakName=tokamakName)
        self._build()

    def _load_config_file(self, tokamakName) -> None:
        """
        Loads the configuration file for the given diagnostic
        """

        pathFileName = self.ws.machine_file(
            tokamakName, "inputs", "sxrInfo", f"{self.configFileName}", required=False
        )

        # --- Load the configuration file, if it exists
        if pathFileName is not None and os.path.isfile(pathFileName):
            self.config_path = pathFileName
            self.info = config_loader(pathFileName)
            if self.info is not None and "NAME" in self.info:
                self.name = self.info["NAME"]
            else:
                self.name = None
                print(
                    f"Configuration file loaded but missing 'NAME' key: {pathFileName}"
                )
            if self.info is not None and "GROUP_NAME" in self.info:
                self.group_name = self.info["GROUP_NAME"]
            else:
                self.group_name = None
                print(
                    f"Configuration file loaded but missing 'GROUP_NAME' key: {pathFileName}"
                )
        else:
            self.info = None
            self.name = None
            self.group_name = None
            print(
                f"Could not load the configuration file '{self.configFileName}' for "
                f"{tokamakName}; searched {[str(r) for r in self.ws.machine_roots(tokamakName)]}"
            )

        # --- Load SubArray configuration files
        if self.info is not None and "SUBARRAY_NAMES" in self.info:
            if len(self.info["SUBARRAY_NAMES"]) > 0:
                for ii, fileName in enumerate(self.info["SUBARRAY_CONFIG_NAMES"]):
                    pathFileName = self.ws.machine_file(
                        tokamakName, "inputs", "sxrInfo", f"{fileName}"
                    )

                    if "SUBARRAYS" not in self.info:
                        self.info["SUBARRAYS"] = {}
                    self.info["SUBARRAYS"][self.info["SUBARRAY_NAMES"][ii]] = (
                        config_loader(str(pathFileName))
                    )

    def _build(self) -> None:
        """
        Definition to build the camera. This will call the correct definition based
        on the data within the bolometer configuration file
        """
        if self.info is not None and "BUILD_TYPE" in self.info:
            if self.info["BUILD_TYPE"] == "FROM PRIMITIVES":
                self._build_from_primitives()
            elif self.info["BUILD_TYPE"] == "KB5V":
                self.load_kb5_camera(camera_id="KB5V", parent=self.world)
            else:
                print(f"Build type of {self.info['BUILD_TYPE']} not yet supported!")
        else:
            print(
                "Bolometer configuration info is missing or incomplete; cannot build camera."
            )

    def _build_from_primitives(self) -> None:
        """
        Builds the bolometer based off the example found here:
        https://www.cherab.info/demonstrations/bolometry/camera_from_primitives.html#bolometer-from-primitives

        Notes:
        If two arrays are close together, it is best to include them within the same
        "bounding box," otherwise some of the raytraced chords might intersect the neighboring
        box

        This will be an array of channels corresponding to the NUM_CHANNELS
        and CHANNEL_TAGS parameters within the sxr configuration file

        The transform command should probably be split off into its own definition, since
        it should be universal
        """

        if self.info is None:
            print(
                "Bolometer configuration info is missing; cannot build from primitives."
            )
            return

        # --- Convenient constants
        XAXIS = Vector3D(1, 0, 0)
        YAXIS = Vector3D(0, 1, 0)
        ZAXIS = Vector3D(0, 0, 1)
        ORIGIN = Point3D(0, 0, 0)

        # --- Constants from the configuration file
        SLIT_WIDTH = self.info["SLIT_WIDTH"]
        SLIT_HEIGHT = self.info["SLIT_HEIGHT"]
        SLIT_THICKNESS = 50.0e-6
        FOIL_WIDTH = self.info["FOIL_WIDTH"]
        FOIL_HEIGHT = self.info["FOIL_HEIGHT"]
        FOIL_CORNER_CURVATURE = self.info["FOIL_CORNER_CURVATURE"]
        SLIT_SENSOR_SEPARATION = self.info["SLIT_SENSOR_SEPARATION"]
        FOIL_SEPARATION = self.info["FOIL_SEPARATION"]
        FOIL_POSITIONS = self.info["FOIL_POSITIONS"]
        CAMERA_POSITION_R_Z_PHI = self.info["CAMERA_POSITION_R_Z_PHI"]
        CAMERA_PHI_RAD = np.deg2rad(CAMERA_POSITION_R_Z_PHI[2])
        x, y = RPhi_To_XY(CAMERA_POSITION_R_Z_PHI[0], CAMERA_PHI_RAD)
        CAMERA_POSITION_X_Y_Z = (x, y, CAMERA_POSITION_R_Z_PHI[1])
        WALL_THICKNESS = 0.005  # 5 mm thick walls

        # --- Derived dimensions
        half_array_width = np.max(
            [
                np.max(np.abs(FOIL_POSITIONS)) * FOIL_SEPARATION + FOIL_WIDTH / 2,
                SLIT_WIDTH / 2,
            ]
        )
        clear_width = 2 * half_array_width
        clear_height = 2 * max(SLIT_HEIGHT, FOIL_HEIGHT)

        housing_width = clear_width + 2 * WALL_THICKNESS
        housing_height = 2 * clear_height + 2 * WALL_THICKNESS
        housing_depth = SLIT_SENSOR_SEPARATION + 2 * FOIL_WIDTH + 2 * WALL_THICKNESS

        # -------------------------------------------------------------------------
        # 1. Create the camera node FIRST so all geometry can be parented to it.
        #    This ensures that any transform on bolometer_camera moves everything.
        # -------------------------------------------------------------------------
        bolometer_camera = BolometerCamera(parent=self.world, name="bolometer_camera")

        # -------------------------------------------------------------------------
        # 2. Build the housing geometry, all parented to bolometer_camera.
        # -------------------------------------------------------------------------
        wall_vec_lower = Vector3D(WALL_THICKNESS, WALL_THICKNESS, WALL_THICKNESS)
        wall_vec_upper = Vector3D(WALL_THICKNESS, WALL_THICKNESS, SLIT_THICKNESS / 2.0)

        outer_lower = Point3D(-housing_width / 2, -housing_height / 2, -housing_depth)
        outer_upper = Point3D(housing_width / 2, housing_height / 2, 0)
        camera_box_outer = Box(
            lower=outer_lower,
            upper=outer_upper,
            parent=bolometer_camera,
            name="Housing Outer",
        )

        # Inner void: inset on all sides by WALL_THICKNESS to leave proper walls
        camera_box_inner = Box(
            lower=outer_lower + wall_vec_lower,
            upper=outer_upper - wall_vec_upper,
            parent=bolometer_camera,
            name="Housing Inner",
        )

        camera_housing = Subtract(
            camera_box_outer,
            camera_box_inner,
            parent=bolometer_camera,
            name="Hollow Housing",
        )

        # Cut the slit aperture through the front face
        aperture = Box(
            lower=Point3D(-SLIT_WIDTH / 2, -SLIT_HEIGHT / 2, -SLIT_THICKNESS / 2),
            upper=Point3D(SLIT_WIDTH / 2, SLIT_HEIGHT / 2, SLIT_THICKNESS / 2),
            parent=bolometer_camera,
            name="Aperture",
        )

        camera_housing = Subtract(
            camera_housing,
            aperture,
            parent=bolometer_camera,
            name="Housing with Aperture",
        )

        camera_housing.material = (
            AbsorbingSurface()
        )  # AbsorbingSurface() or NullMaterial()

        # Attach the finished housing to the camera
        bolometer_camera.camera_geometry = camera_housing

        # --- Create a slit at the camera origin
        slit = BolometerSlit(
            slit_id=f"{self.info['NAME']} slit",
            centre_point=ORIGIN,
            basis_x=XAXIS,
            dx=SLIT_WIDTH,
            basis_y=YAXIS,
            dy=SLIT_HEIGHT,
            dz=SLIT_THICKNESS,
            parent=bolometer_camera,
            csg_aperture=False,  # This messes the JET bolometers up if it is set to True
        )

        # --- Create the sensor node behind the slit
        sensor = Node(
            name=f"{self.info['NAME']} sensor",
            parent=bolometer_camera,
            transform=translate(0, 0, -SLIT_SENSOR_SEPARATION),
        )

        # --- Create the foils relative to the sensor
        for ii, shift in enumerate(FOIL_POSITIONS):

            # Older version
            foil_transform = translate(shift * FOIL_SEPARATION, 0, 0) * sensor.transform
            foil = BolometerFoil(
                detector_id=self.info["CHANNEL_TAGS"][ii],
                centre_point=ORIGIN.transform(foil_transform),
                basis_x=XAXIS.transform(foil_transform),
                dx=FOIL_WIDTH,
                basis_y=YAXIS.transform(foil_transform),
                dy=FOIL_HEIGHT,
                slit=slit,
                parent=bolometer_camera,
                accumulate=False,
                curvature_radius=FOIL_CORNER_CURVATURE,
                units="Radiance",  # Default units, changed during radDist.bolos_observe() to match the radDist config file
            )

            bolometer_camera.add_foil_detector(foil)

        # --- Translate the camera to the correct position
        origin_xyz = Vector3D(*CAMERA_POSITION_X_Y_Z)

        # --- Tilt the camera downward if it is downward facing
        sign = -1 if self.info["CAMERA_DOWNWARD_FACING"] else 1
        tilt_rad = sign * np.deg2rad(self.info["CAMERA_ROTATION"])

        e_R = Vector3D(np.cos(CAMERA_PHI_RAD), np.sin(CAMERA_PHI_RAD), 0).normalise()
        e_z = ZAXIS
        e_phi = e_z.cross(e_R).normalise()

        # Rotate the bolometer if the chords extend toroidally instead of the typical poloidal fan array
        skew_rad = 0.0
        if "SKEW_ANGLE" in self.info:
            skew_rad = np.deg2rad(self.info["SKEW_ANGLE"])

        e_phi_skewed = rotate_vector(e_phi, e_z, skew_rad).normalise()

        # Rotate e_z in R–z plane for poloidal tilt
        view_dir = rotate_vector(e_z, e_phi_skewed, tilt_rad).normalise()

        # Build orthonormal basis
        z_axis = view_dir  # Camera z-axis: the direction the camera "looks"
        y_axis = e_phi  # Camera y-axis: e_phi (toroidal direction)
        # Camera x-axis: y × z to form right-handed coordinate system
        x_axis = y_axis.cross(z_axis).normalise()
        # --- Force orthogonality for safety
        y_axis = z_axis.cross(x_axis).normalise()

        # Build full rotation matrix from orthonormal basis
        basis_matrix = AffineMatrix3D(
            [
                x_axis.x,
                y_axis.x,
                z_axis.x,
                origin_xyz.x,
                x_axis.y,
                y_axis.y,
                z_axis.y,
                origin_xyz.y,
                x_axis.z,
                y_axis.z,
                z_axis.z,
                origin_xyz.z,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        )

        # Construct transform matrix
        bolometer_camera.transform = basis_matrix

        self.bolometer_camera = bolometer_camera

    def _calc_etendues(self) -> None:
        """
        Calculate the etendue based on the geometery of the camera.
        Taken from the Cherab demo:
        https://www.cherab.info/demonstrations/bolometry/calculate_etendue.html

        NOTE: If the ETENDUE is in the config file, the program will use that instead
        of any calculated values
        """

        if self.info is None:
            print(
                "Bolometer configuration info is missing; cannot build from primitives."
            )
            return

        if "ETENDUE" in self.info:
            self.etendues = self.info["ETENDUE"]
            self.etendues_error = (
                self.info["ETENDUE_ERROR"]
                if "ETENDUE_ERROR" in self.info
                else [0.0] * len(self.etendues)
            )
            return

        if "FOIL_SLIT_ANGLE_FACTOR" in self.info:
            FOIL_SLIT_ANGLE_FACTOR = self.info["FOIL_SLIT_ANGLE_FACTOR"]
        else:
            FOIL_SLIT_ANGLE_FACTOR = 1.0

        self.etendues = []
        self.etendues_error = []
        analytic_etendues = []
        for foil in self.bolometer_camera:
            raytraced_etendue, raytraced_error = foil.calculate_etendue(
                ray_count=10_000
            )
            Adet = foil.x_width * foil.y_width
            Aslit = foil.slit.dx * foil.slit.dy
            costhetadet = foil.sightline_vector.normalise().dot(foil.normal_vector)
            costhetaslit = foil.sightline_vector.normalise().dot(
                foil.slit.normal_vector
            )
            distance = foil.centre_point.vector_to(foil.slit.centre_point).length
            analytic_etendue = Adet * Aslit * costhetadet * costhetaslit / distance**2
            """
            print(
                "{} raytraced etendue: {:.4g} +- {:.1g} analytic: {:.4g}".format(
                    foil.name, raytraced_etendue, raytraced_error, analytic_etendue
                )
            )
            """
            raytraced_etendue = raytraced_etendue * FOIL_SLIT_ANGLE_FACTOR
            analytic_etendue = analytic_etendue * FOIL_SLIT_ANGLE_FACTOR
            self.etendues.append(raytraced_etendue.item())
            self.etendues_error.append(raytraced_error.item())
            analytic_etendues.append(analytic_etendue)
        self.etendues_analytic = analytic_etendues
        self.etendues_analytic_error = (np.array(analytic_etendues) * 0.1).tolist()

    def _change_parent(self, value=None) -> None:
        """
        Either sets the self.bolometer_camera.parent to None or self.world
        """

        self.bolometer_camera.parent = value

    def load_kb5_camera(self, camera_id, parent=None) -> None:
        """
        Loads the KB5 camera configuration from the locally stored csv file. The csv file can
        be found here: https://github.com/cherab/jet/tree/master/cherab/jet/bolometry/kb5

        Add DATA_PATH to the bolometer configuration file to specify the location of the csv files.

        Also, change the BUILD_TYPE to "KB5V" in the configuration file
        """

        if not self.info:
            raise ValueError("No info loaded for KB5 camera.")

        if "DATA_PATH" not in self.info:
            raise ValueError("No DATA_PATH in info for KB5 camera.")

        # A relative DATA_PATH is relative to the camera's configuration file
        _DATA_PATH = os.path.expanduser(str(self.info["DATA_PATH"]))
        if not os.path.isabs(_DATA_PATH) and self.config_path is not None:
            _DATA_PATH = os.path.join(os.path.dirname(self.config_path), _DATA_PATH)

        if camera_id == "KB5V":
            foils = np.loadtxt(
                os.path.join(_DATA_PATH, "kb5v_foils.csv"), delimiter=","
            )
            slits = np.loadtxt(
                os.path.join(_DATA_PATH, "kb5v_slits.csv"), delimiter=","
            )
        elif camera_id == "KB5H":
            foils = np.loadtxt(
                os.path.join(_DATA_PATH, "kb5h_foils.csv"), delimiter=","
            )
            slits = np.loadtxt(
                os.path.join(_DATA_PATH, "kb5h_slits.csv"), delimiter=","
            )
        else:
            raise ValueError("Unrecognised bolometer camera_id '{}'.".format(camera_id))

        num_slits = slits.shape[0]
        num_foils = foils.shape[0]

        bolometer_camera = BolometerCamera(name=camera_id, parent=parent)

        slit_objects = {}
        for i in range(num_slits):
            slit_data = slits[i]
            slit_id = "{}_Slit_#{}".format(camera_id, str(int(slit_data[0])))
            p1 = Point3D(slit_data[1], slit_data[2], slit_data[3])
            p2 = Point3D(slit_data[4], slit_data[5], slit_data[6])
            p3 = Point3D(slit_data[7], slit_data[8], slit_data[9])
            p4 = Point3D(slit_data[10], slit_data[11], slit_data[12])
            basis_x = p1.vector_to(p2).normalise()
            dx = p1.distance_to(p2)
            basis_y = p2.vector_to(p3).normalise()
            dy = p2.distance_to(p3)
            centre_point = Point3D(
                (p1.x + p2.x + p3.x + p4.x) / 4,
                (p1.y + p2.y + p3.y + p4.y) / 4,
                (p1.z + p2.z + p3.z + p4.z) / 4,
            )
            slit_objects[slit_id] = BolometerSlit(
                slit_id, centre_point, basis_x, dx, basis_y, dy, parent=bolometer_camera
            )

        for i in range(num_foils):
            foil_data = foils[i]
            foil_id = "{}_CH{}_Foil".format(camera_id, str(int(foil_data[0])))
            slit_id = "{}_Slit_#{}".format(camera_id, str(int(foil_data[1])))

            p1 = Point3D(foil_data[2], foil_data[3], foil_data[4])
            p2 = Point3D(foil_data[5], foil_data[6], foil_data[7])
            p3 = Point3D(foil_data[8], foil_data[9], foil_data[10])
            p4 = Point3D(foil_data[11], foil_data[12], foil_data[13])
            basis_x = p2.vector_to(
                p1
            ).normalise()  # switching orientation to ensure face orientation is correct
            dx = p1.distance_to(p2)
            basis_y = p2.vector_to(p3).normalise()
            dy = p2.distance_to(p3)
            centre_point = Point3D(
                (p1.x + p2.x + p3.x + p4.x) / 4,
                (p1.y + p2.y + p3.y + p4.y) / 4,
                (p1.z + p2.z + p3.z + p4.z) / 4,
            )

            # Shift backwards 3mm for all foils except those explicitly measured on back plate
            if camera_id == "KB5V":
                if i not in (9 - 1, 25 - 1, 32 - 1):
                    basis_z = basis_x.cross(basis_y).normalise()
                    centre_point = centre_point - basis_z * 0.0032
                # if i == 9 - 1:
                #     centre_point = centre_point + basis_x * 0.001
                if i == 31 - 1:
                    centre_point = centre_point - basis_x * 0.001
            else:
                basis_z = basis_x.cross(basis_y).normalise()
                centre_point = centre_point - basis_z * 0.0032

            foil = BolometerFoil(
                foil_id,
                centre_point,
                basis_x,
                dx,
                basis_y,
                dy,
                slit_objects[slit_id],
                parent=bolometer_camera,
            )

            bolometer_camera.add_foil_detector(foil)

        self.bolometer_camera = bolometer_camera

    def change_camera_material(self, material=""):
        """
        Changes the 'Housing with Aperture' material to the
        input material
        """

        if material not in ["Absorbing", "Null"]:
            raise RuntimeError("Material input must be Absorbing or Null")

        mat = AbsorbingSurface()
        if material == "Null":
            mat = NullMaterial()

        for c_ in self.bolometer_camera.children:
            if c_.name == "Housing with Aperture":
                c_.material = mat
