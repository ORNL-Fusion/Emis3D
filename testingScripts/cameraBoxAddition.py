# cameraBoxAddition.py
"""
Testing creting a box around a bolometer to have more accurate etendue calculations
"""


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from emis3d.Diagnostic import Bolometer
from cherab.tools.observers import BolometerCamera, BolometerFoil, BolometerSlit
from emis3d.Tokamak import Tokamak
from raysect.core import Point3D, Node, translate, rotate_basis, Point2D
from emis3d.Util import (
    point3d_to_rz,
    get_rectangle_corners,
    compute_etendue_metric,
)
from emis3d.Util_plotting import draw_Cherab_box, get_to_world, extract_csg_bounds
from raysect.optical import World
from raysect.primitive import Box, Subtract
from raysect.optical import AbsorbingSurface, NullMaterial  # type:ignore
from raysect.core.math import Vector3D


tokamakName = "DIII-D"
configFileName = "SX90PF_UP_config.yaml"

bolo = None
try:
    bolo = Bolometer(world=None, tokamakName=tokamakName, configFileName=configFileName)
except Exception as e:
    print(f"An error occured, {e}")

plt.ion()


"""
The camera is constructed with the slit plane at z = 0, facing in the positive z-direction

"""

# Camera is built in the diagnostics class
if False:
    t = Tokamak(
        tokamakName="DIII-D",
        mode="Analysis",
        reflections=False,
        loadBolometers=True,
    )

    bolometer_camera = t.bolometers[0].bolometer_camera
    foil = bolometer_camera.foil_detectors[0]

    # t.bolometers[0].change_camera_material(material="Absorbing")
    print(foil.calculate_etendue())
    # t.bolometers[1].change_camera_material(material="Absorbing")
    # print("After changing to absorbing", foil.calculate_etendue())
    # t.bolometers[1].change_camera_material(material="Null")
    # print("After changing back to Null", foil.calculate_etendue())
    # Housing
    # draw_box(ax, inner_lower, inner_upper)
    # draw_box(ax, slit_lower, slit_upper)

    if True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = ["black", "red", "green", "blue"]
        # TODO: Make it match the name = 'Camera', "slit", etc.
        draw_Cherab_box(
            ax, bolometer_camera, to_world=True
        )  # NOTE the camera geometery does not translate correctly!!!

        for ii, foil in enumerate(bolometer_camera):
            # if ax is not None:
            length = 0.1
            debug = True
            slit = foil.slit

            if debug:

                foil_center = foil.centre_point
                slit_center = slit.to_root() * slit.centre_point

                separation = np.sqrt(
                    (foil_center.x - slit_center.x) ** 2
                    + (foil_center.y - slit_center.y) ** 2
                    + (foil_center.z - slit_center.z) ** 2
                )

                print("\n--- FOIL to SLIT SEPARATION ---")
                print(f"Center-to-center distance: {separation:.6f} m")

            # Get corners
            foil_corners = get_rectangle_corners(foil)
            slit_corners = get_rectangle_corners(slit)

            # Identify upper/lower in Z
            foil_sorted = sorted(foil_corners, key=lambda p: p.z)
            slit_sorted = sorted(slit_corners, key=lambda p: p.z)

            foil_lower = foil_sorted[0]
            foil_upper = foil_sorted[-2]
            slit_lower = slit_sorted[0]
            slit_upper = slit_sorted[-2]

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
            corners = []
            for p in foil_corners:
                corners.append([p.x, p.y, p.z])
                # r, z = point3d_to_rz(p)
                ax.scatter(*p, color="black")
            corners2 = []
            corners2.append(corners[0])
            corners2.append(corners[1])
            corners2.append(corners[3])
            corners2.append(corners[2])

            poly = Poly3DCollection([corners2], label=foil.name)
            ax.add_collection3d(poly)

            corners = []
            for p in slit_corners:
                corners.append([p.x, p.y, p.z])
                # r, z = point3d_to_rz(p)
                ax.scatter(*p, color="purple")
            corners2 = []
            corners2.append(corners[0])
            corners2.append(corners[1])
            corners2.append(corners[3])
            corners2.append(corners[2])

            poly = Poly3DCollection([corners2], color="purple")
            ax.add_collection3d(poly)


# Building the camera here
if True:
    world = World()
    # --- Convenient constants
    XAXIS = Vector3D(1, 0, 0)
    YAXIS = Vector3D(0, 1, 0)
    ZAXIS = Vector3D(0, 0, 1)
    ORIGIN = Point3D(0, 0, 0)

    if bolo is not None:
        if bolo.info is not None:

            # --- Input parameters
            SLIT_WIDTH = bolo.info["SLIT_WIDTH"]  # Built on x-axis
            SLIT_HEIGHT = bolo.info["SLIT_HEIGHT"]  # Built on the y-axis
            SLIT_THICKNESS = 0.0005
            FOIL_WIDTH = bolo.info["FOIL_WIDTH"]  # Width of the diode
            FOIL_HEIGHT = bolo.info["FOIL_HEIGHT"]  # Height of the diode
            FOIL_SEPARATION = bolo.info["FOIL_SEPARATION"]
            FOIL_POSITIONS = bolo.info["FOIL_POSITIONS"]
            SLIT_SENSOR_SEPARATION = bolo.info["SLIT_SENSOR_SEPARATION"]
            FOIL_CORNER_CURVATURE = bolo.info["FOIL_CORNER_CURVATURE"]
            WALL_THICKNESS = 0.005  # 5 mm thick walls

            # --- Derived dimensions
            half_array_width = (
                np.max(np.abs(FOIL_POSITIONS)) * FOIL_SEPARATION + FOIL_WIDTH / 2
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
            bolometer_camera = BolometerCamera(parent=world, name="bolometer_camera")

            # -------------------------------------------------------------------------
            # 2. Build the housing geometry, all parented to bolometer_camera.
            # -------------------------------------------------------------------------
            wall_vec_lower = Vector3D(WALL_THICKNESS, WALL_THICKNESS, WALL_THICKNESS)
            wall_vec_upper = Vector3D(
                WALL_THICKNESS, WALL_THICKNESS, SLIT_THICKNESS / 2.0
            )

            outer_lower = Point3D(
                -housing_width / 2, -housing_height / 2, -housing_depth
            )
            outer_upper = Point3D(housing_width / 2, housing_height / 2, 0)

            camera_box_outer = Box(
                lower=outer_lower,
                upper=outer_upper,
                parent=world,
                name="Housing Outer",
            )

            # Inner void: inset on all sides by WALL_THICKNESS to leave proper walls
            camera_box_inner = Box(
                lower=outer_lower + wall_vec_lower,
                upper=outer_upper - wall_vec_upper,
                parent=world,
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
            camera_housing.material = NullMaterial()

            # Attach the finished housing to the camera
            bolometer_camera.camera_geometry = camera_housing

            # -------------------------------------------------------------------------
            # 3. Define the slit (in bolometer_camera local coordinates)
            # -------------------------------------------------------------------------
            slit = BolometerSlit(
                slit_id="slit",
                centre_point=ORIGIN,
                basis_x=XAXIS,
                basis_y=YAXIS,
                dx=SLIT_WIDTH,
                dy=SLIT_HEIGHT,
                dz=SLIT_THICKNESS,
                parent=bolometer_camera,
            )

            # -------------------------------------------------------------------------
            # 4. Sensor node and foil detectors
            # -------------------------------------------------------------------------
            sensor = Node(
                name="sensor",
                parent=bolometer_camera,
                transform=translate(0, 0, -SLIT_SENSOR_SEPARATION),
            )

            # --- Create the foils relative to the sensor
            for ii, shift in enumerate(FOIL_POSITIONS):

                # Older version
                foil_transform = (
                    translate(shift * FOIL_SEPARATION, 0, 0) * sensor.transform
                )
                foil = BolometerFoil(
                    detector_id=bolo.info["CHANNEL_TAGS"][ii],
                    centre_point=ORIGIN.transform(foil_transform),
                    basis_x=XAXIS.transform(foil_transform),
                    dx=FOIL_WIDTH,
                    basis_y=YAXIS.transform(foil_transform),
                    dy=FOIL_HEIGHT,
                    slit=slit,
                    parent=bolometer_camera,
                    units="Power",
                    accumulate=False,
                    curvature_radius=FOIL_CORNER_CURVATURE,
                )

                bolometer_camera.add_foil_detector(foil)

            bolometer_camera.transform = translate(0, 0, 1) * rotate_basis(
                -ZAXIS, 0.5 * YAXIS
            )

            # -----------------------------
            # Plot everything
            # -----------------------------

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            # Housing
            # draw_box(ax, inner_lower, inner_upper)
            # draw_box(ax, slit_lower, slit_upper)

            # TODO: Make it match the name = 'Camera', "slit", etc.

            draw_Cherab_box(ax, bolometer_camera, to_world=True)

            # Slit
            # draw_Cherab_box(
            #    ax,
            #    bolometer_camera.children[1].children[0],
            #    colors=["purple"],
            # )

            # Foil
            """
            draw_Cherab_box(
                ax,
                bolometer_camera.children[4].children[0],
                colors=["black"],
            )
            """
            # colors = ["black", "red", "green", "blue"]
            for ii, foil in enumerate(bolometer_camera):
                # if ax is not None:
                length = 0.1
                debug = True

                if debug:

                    foil_center = foil.centre_point
                    slit_center = slit.centre_point

                    separation = np.sqrt(
                        (foil_center.x - slit_center.x) ** 2
                        + (foil_center.y - slit_center.y) ** 2
                        + (foil_center.z - slit_center.z) ** 2
                    )

                    print("\n--- FOIL to SLIT SEPARATION ---")
                    print(f"Center-to-center distance: {separation * 100.0:.2f} cm")

                # Get corners
                foil_corners = get_rectangle_corners(foil)
                slit_corners = get_rectangle_corners(slit)

                # Identify upper/lower in Z
                foil_sorted = sorted(foil_corners, key=lambda p: p.z)
                slit_sorted = sorted(slit_corners, key=lambda p: p.z)

                foil_lower = foil_sorted[0]
                foil_upper = foil_sorted[-2]
                slit_lower = slit_sorted[0]
                slit_upper = slit_sorted[-2]

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
                corners = []
                for p in foil_corners:
                    corners.append([p.x, p.y, p.z])
                    # r, z = point3d_to_rz(p)
                    ax.scatter(*p, color="black")
                corners2 = []
                corners2.append(corners[0])
                corners2.append(corners[1])
                corners2.append(corners[3])
                corners2.append(corners[2])

                poly = Poly3DCollection([corners2], color="tab:blue", label=foil.name)
                ax.add_collection3d(poly)

                corners = []
                for p in slit_corners:
                    corners.append([p.x, p.y, p.z])
                    # r, z = point3d_to_rz(p)
                    ax.scatter(*p, color="purple")
                corners2 = []
                corners2.append(corners[0])
                corners2.append(corners[1])
                corners2.append(corners[3])
                corners2.append(corners[2])

                poly = Poly3DCollection([corners2], color="purple")
                ax.add_collection3d(poly)

                # Plot envelope lines
                if False:
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

            ax.set_xlabel("X (foil array direction)")
            ax.set_ylabel("Y (toroidal)")
            ax.set_zlabel("Z (view direction)")
            ax.set_box_aspect([1, 1, 1])
            ax.legend()

            """
            # Foils

            for centre in foil_centres:
                cx, cy, cz = centre

                corners = np.array(
                    [
                        [cx - FOIL_WIDTH / 2, cy - FOIL_HEIGHT / 2, cz],
                        [cx + FOIL_WIDTH / 2, cy - FOIL_HEIGHT / 2, cz],
                        [cx + FOIL_WIDTH / 2, cy + FOIL_HEIGHT / 2, cz],
                        [cx - FOIL_WIDTH / 2, cy + FOIL_HEIGHT / 2, cz],
                    ]
                )

                poly = Poly3DCollection([corners])
                ax.add_collection3d(poly)

                # Foil normal (pointing toward slit, -z direction)
                ax.quiver(cx, cy, cz, 0, 0, -1)

            # Sensor node
            ax.scatter(sensor_node[0], sensor_node[1], sensor_node[2])

            # Arrow from sensor to slit
            slit_centre = np.array([0, 0, WALL_THICKNESS / 2])
            arrow_vec = slit_centre - sensor_node

            ax.quiver(
                sensor_node[0],
                sensor_node[1],
                sensor_node[2],
                arrow_vec[0],
                arrow_vec[1],
                arrow_vec[2],
            )


            ax.set_xlabel("X (foil array direction)")
            ax.set_ylabel("Y (toroidal)")
            ax.set_zlabel("Z (view direction)")
            ax.set_box_aspect([1, 1, 1])
            ax.legend()

            plt.show()
            """

    """

    raytraced_etendues = []
    raytraced_errors = []
    analytic_etendues = []
    for foil in bolometer_camera:
        raytraced_etendue, raytraced_error = foil.calculate_etendue(ray_count=100_000)
        Adet = foil.x_width * foil.y_width
        Aslit = foil.slit.dx * foil.slit.dy
        costhetadet = foil.sightline_vector.normalise().dot(foil.normal_vector)
        costhetaslit = foil.sightline_vector.normalise().dot(foil.slit.normal_vector)
        distance = foil.centre_point.vector_to(foil.slit.centre_point).length
        analytic_etendue = Adet * Aslit * costhetadet * costhetaslit / distance**2
        print(
            "{} raytraced etendue: {:.4g} +- {:.1g} analytic: {:.4g}".format(
                foil.name, raytraced_etendue, raytraced_error, analytic_etendue
            )
        )
        raytraced_etendues.append(raytraced_etendue)
        raytraced_errors.append(raytraced_error)
        analytic_etendues.append(analytic_etendue)
    etendues = analytic_etendues
    etendues_error = np.array(analytic_etendues) * 0.1
    """
