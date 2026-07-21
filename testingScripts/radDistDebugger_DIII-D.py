# radDistDebugger_DIII-D.py
"""
Manually input x, y, z points for a specific radDist, see what the bolometer observes

"""

# radDistTester_DIII-D.py
"""
This program will group similar SXR arrays, then plot out
the chords, radDist contour plot, and the observed radiation
below.

It is currently specific to DIII-D
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np

import main.Util_radDist as Util_radDist
import main.radDist as radDist
from main.Globals import *
from main.Tokamak import Tokamak
from main.Util import config_loader, point3d_to_rz
from raysect.optical import VolumeTransform  # type: ignore
from cherab.tools.emitters import RadiationFunction
from raysect.core import Point3D

tokamakName = "DIII-D"
configFileName = "sqaureTube_config.yaml"  # "sqaureTube_config.yaml"  # "elongatedRing_config.yaml"  # "helical_config.yaml"  #
sigma_z = 0.1
sigma_R = 0.1
rotationAngle = 0.0
rzvalues = [1.65, 0.0]
# Only use a specific channel
name = "SX45F_DOWN"
channel = "SX45F07"


def sample_bolometer_foil(foil, resolution=20):

    # Foil local dimensions
    w = foil.y_width / 2
    h = foil.x_width / 2

    xs = np.linspace(-w, w, resolution)
    ys = np.linspace(-h, h, resolution)

    sampled_points = []

    for x in xs:
        for y in ys:

            # Local point on foil
            local_point = Point3D(x, y, 0)

            # Convert to world space
            world_point = foil.to_root() * local_point

            # Store it
            sampled_points.append(world_point)

    return sampled_points


# --- Create the radDist using only one point, we don't need to loop over everything
pathFileName = os.path.join(
    EMIS3D_INPUTS_DIRECTORY, tokamakName, "radDists", configFileName
)
config = config_loader(pathFileName)
if config is None:
    raise FileNotFoundError(f"Could not load config file: {pathFileName}")


# --- Update the configuration file
rzArray = [rzvalues[0], rzvalues[1]]


# --- Create the radDist
if config["distType"] == "Helical":
    rD = radDist.Helical(startR=rzArray[0], startZ=rzArray[1], config=config)
elif config["distType"] == "ElongatedRing":
    rD = radDist.ElongatedRing(startR=rzArray[0], startZ=rzArray[1], config=config)
elif config["distType"] == "SquareTube":
    rD = radDist.SquareTube(startR=rzArray[0], startZ=rzArray[1], config=config)
else:
    raise RuntimeError(
        "Please have 'elongatedRing', 'helical', or 'SqureTube' in the configFileName"
    )
rD.emissionName = rD.info["emissionNames"][0]

# --- Make up a bunch of points and see what this does:
rD._evaluate([1.1, 1.2], [0.5, 0.5], [0, 0])

# --- Test emission properties with cherab

# Change emission surface material
emitter = None
for val in rD.tokamak.world.children:
    if val.name == "Emission Surface":
        emitter = val

if emitter is not None:
    emittting_material = VolumeTransform(
        RadiationFunction(rD._evaluate_cherab), emitter.transform.inverse()
    )
    emitter.material = emittting_material


bolo = None
foil = None
for b_ in rD.tokamak.bolometers:
    if b_.name == name:
        bolo = b_.bolometer_camera

if bolo is not None:
    for f_ in bolo.foil_detectors:
        if f_.name == channel:
            foil = f_
            foil.pixel_samples = 40


"""
if foil is not None:
    foil.units = "Power"
    foil.observe()

    print(f"Channel: {channel}")
    print(
        f"Observed power {foil.pipelines[0].value.mean} +/- {foil.pipelines[0].value.error()}"
    )
"""

points = sample_bolometer_foil(foil, resolution=30)


if True:
    f = plt.figure()
    ax = f.add_subplot(111)
    for p_ in points:
        p = p_[0]
        r, z = point3d_to_rz(p)
        ax.plot(r, z, "r.", markersize=2)

    plt.show()


# --- Plot everything ---
# rD.plotOverview(plot_etendue=["SX45F07"])
