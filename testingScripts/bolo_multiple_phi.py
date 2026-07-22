# bolo_multiple_phi.py
"""
Test script that has a bolometer view different phi regions at once
and bin them correctly.

Skip to HERE for the start of the program, the top is just from:
https://www.cherab.info/demonstrations/bolometry/observing_radiation_function.html#bolometer-observing-radiation

See also:
https://www.cherab.info/demonstrations/line_emission/custom_emitter.html

"""

import matplotlib.pyplot as plt
import numpy as np
from raysect.core import Node, Point3D, Vector3D, rotate_basis, translate
from raysect.optical import World
from raysect.optical.material import (
    AbsorbingSurface,
    VolumeTransform,
    InhomogeneousVolumeEmitter,
    NumericalIntegrator,
)
from raysect.primitive import Box, Cylinder, Subtract
from raysect.optical.material.emitter import InhomogeneousVolumeEmitter
from cherab.core.math import AxisymmetricMapper, sample2d
from cherab.tools.emitters import RadiationFunction
from cherab.tools.observers.bolometry import (
    BolometerCamera,
    BolometerSlit,
    BolometerFoil,
)
from raysect.optical.observer import PowerPipeline0D, SpectralPowerPipeline0D

# Convenient constants
XAXIS = Vector3D(1, 0, 0)
YAXIS = Vector3D(0, 1, 0)
ZAXIS = Vector3D(0, 0, 1)
ORIGIN = Point3D(0, 0, 0)
# Bolometer geometry
BOX_WIDTH = 0.05
BOX_HEIGHT = 0.07
BOX_DEPTH = 0.2
SLIT_WIDTH = 0.004
SLIT_HEIGHT = 0.005
FOIL_WIDTH = 0.0013
FOIL_HEIGHT = 0.0038
FOIL_CORNER_CURVATURE = 0.0005
SLIT_SENSOR_SEPARATION = 0.1
FOIL_SEPARATION = 0.00508  # 0.2 inch between foils

world = World()

########################################################################
# Build a simple bolometer camera.
########################################################################

# The camera consists of a box with a rectangular slit and 4 foils.
# In its local coordinate system, the camera's slit is located at the
# origin and the foils below the z=0 plane, looking up towards the slit.
camera_box = Box(
    lower=Point3D(-BOX_WIDTH / 2, -BOX_HEIGHT / 2, -BOX_DEPTH),
    upper=Point3D(BOX_WIDTH / 2, BOX_HEIGHT / 2, 0),
)
# Hollow out the box
inside_box = Box(
    lower=camera_box.lower + Vector3D(1e-5, 1e-5, 1e-5),
    upper=camera_box.upper - Vector3D(1e-5, 1e-5, 1e-5),
)
camera_box = Subtract(camera_box, inside_box)
# The slit is a hole in the box
aperture = Box(
    lower=Point3D(-SLIT_WIDTH / 2, -SLIT_HEIGHT / 2, -1e-4),
    upper=Point3D(SLIT_WIDTH / 2, SLIT_HEIGHT / 2, 1e-4),
)
camera_box = Subtract(camera_box, aperture)
camera_box.material = AbsorbingSurface()
# Instance of the bolometer camera
bolometer_camera = BolometerCamera(camera_geometry=camera_box)
# The bolometer slit in this instance just contains targeting information
# for the ray tracing, since we have already given our camera a geometry
# The slit is defined in the local coordinate system of the camera
slit = BolometerSlit(
    slit_id="Example slit",
    centre_point=ORIGIN,
    basis_x=XAXIS,
    dx=SLIT_WIDTH,
    basis_y=YAXIS,
    dy=SLIT_HEIGHT,
    parent=bolometer_camera,
    csg_aperture=False,
)
# 4 bolometer foils, spaced at equal intervals along the local X axis
# The bolometer positions and orientations are given in the local coordinate
# system of the camera, just like the slit
sensor = Node(
    name="Bolometer sensor",
    parent=bolometer_camera,
    transform=translate(0, 0, -SLIT_SENSOR_SEPARATION),
)
# The foils are shifted relative to the centre of the sensor by -1.5, -0.5, 0.5 and 1.5
# times the foil-foil separation
for i, shift in enumerate([-1.5, -0.5, 0.5, 1.5]):
    foil_transform = translate(shift * FOIL_SEPARATION, 0, 0) * sensor.transform
    foil = BolometerFoil(
        detector_id="Foil {}".format(i + 1),
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

bolometer_camera.transform = translate(1, 0, 1.5) * rotate_basis(-ZAXIS, YAXIS)
bolometer_camera.parent = world

########################################################################
# HERE HERE HERE HERE HERE HERE HERE
########################################################################

# The plasma will be a cylindrical plasma which emits with a constant
# emissivity of 1 W/m3 in an annular ring.
# See the RadiationFunction example for more details of how this is set up.
MAJOR_RADIUS = 1
MINOR_RADIUS = 0.5
CENTRE_Z = -0.5
CYLINDER_HEIGHT = 5
EMISSIVITY = 1

N_TOROIDAL_BINS = 2
WL_MIN = 300
WL_MAX = 500
WAVELENGTHS = [350, 450]


def emission_function_full(r, z):
    if (r - MAJOR_RADIUS) ** 2 + (z - CENTRE_Z) ** 2 < MINOR_RADIUS**2:
        return EMISSIVITY
    return 0


def emission_function_top(r, z):
    if (r - MAJOR_RADIUS) ** 2 + (z - CENTRE_Z) ** 2 < MINOR_RADIUS**2:
        if z >= CENTRE_Z:
            return EMISSIVITY
    return 0


def emission_function_bottom(r, z):
    if (r - MAJOR_RADIUS) ** 2 + (z - CENTRE_Z) ** 2 < MINOR_RADIUS**2:
        if z < CENTRE_Z:
            return EMISSIVITY
    return 0


class ExcitationLines(InhomogeneousVolumeEmitter):
    def __init__(self, step=0.005):
        super().__init__(NumericalIntegrator(step=step))

    def emission_function(
        self, point, direction, spectrum, world, ray, primitive, to_local, to_world
    ):

        x, y, z = point.transform(to_world)
        r = np.sqrt(x**2 + y**2)

        if (r - MAJOR_RADIUS) ** 2 + (z - CENTRE_Z) ** 2 > MINOR_RADIUS**2:
            return spectrum

        print(spectrum.wavelengths)
        return spectrum


emitter = Cylinder(
    radius=MAJOR_RADIUS + MINOR_RADIUS,
    height=CYLINDER_HEIGHT,
    transform=translate(0, 0, CENTRE_Z - CYLINDER_HEIGHT / 2),
    parent=world,
)

########################################################################
# 1. Measure everything at once as a sanity check
########################################################################
emission_function_3d = AxisymmetricMapper(emission_function_full)
emitter.material = VolumeTransform(
    RadiationFunction(emission_function_3d), transform=emitter.transform.inverse()
)
data = {}
data["SANITY CHECK"] = []
for foil in bolometer_camera:
    # Ensure reasonable sampling statistics
    foil.pixel_samples = 100000
    # Measure the incident power
    foil.units = "Power"
    foil.observe()
    data["SANITY CHECK"].append(foil.pipelines[0].value.mean)


########################################################################
# Measure the top and bottom seperate
########################################################################
emission_function_3d = AxisymmetricMapper(emission_function_top)
emitter.material = VolumeTransform(
    RadiationFunction(emission_function_3d), transform=emitter.transform.inverse()
)


data["TOP ONLY"] = []
for foil in bolometer_camera:
    # Ensure reasonable sampling statistics
    foil.pixel_samples = 100000
    # Measure the incident power
    foil.units = "Power"
    foil.observe()
    data["TOP ONLY"].append(foil.pipelines[0].value.mean)


emission_function_3d = AxisymmetricMapper(emission_function_bottom)
emitter.material = VolumeTransform(
    RadiationFunction(emission_function_3d), transform=emitter.transform.inverse()
)

data["BOTTOM ONLY"] = []
for foil in bolometer_camera:
    # Ensure reasonable sampling statistics
    foil.pixel_samples = 100000
    # Measure the incident power
    foil.units = "Power"
    foil.observe()
    data["BOTTOM ONLY"].append(foil.pipelines[0].value.mean)


########################################################################
# Measure both at the same time, using wavelength to differentiate top and bottom
########################################################################
"""
The overall point is to have the values above CENTRE_Z emit at 450 nm
and the values below CENTRE_Z emit at 350 nm, then combine them for the 
total power

emitter.material = ExcitationLines()

spectral_pipeline = SpectralPowerPipeline0D(display_progress=False)

foil.pipelines = [spectral_pipeline]
"""

########################################################################
# Print the results
########################################################################
for ii in range(4):
    tb = data["TOP ONLY"][ii] + data["BOTTOM ONLY"][ii]
    print(
        f"Foil {ii}: Sanity Check: {data['SANITY CHECK'][ii]:.2e}, Top + Bottom {tb:.2e}"
    )


# Insert nice figure here
