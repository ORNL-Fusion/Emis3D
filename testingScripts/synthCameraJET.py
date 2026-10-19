# synthCamera.py
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
from emis3d.Tokamak import Tokamak
import pdb


tokamakName = "JET"

tok = Tokamak(
    tokamakName="JET",
    mode="Analysis",
    reflections=False,
    loadBolometers=True,
    )

#breakpoint()

# tok.synth_camera_test(CameraType=1, TorAngleDeg=-135, Title="29_closer",\
#                       IndLightSize=0.01, BoundingCylMult=1.5, ZoomMult=1.65,\
#                       Zadjust=0.25, Radjust=0.0,\
#                       WithWallCAD=False, SpecBins=10, PixelSamples=100)

# tok.synth_camera_test(CameraType=0, TorAngleDeg=-135, Title="51_outofworld",\
#                       IndLightSize=0.001, BoundingCylMult=1.5, ZoomMult=0.5,\
#                       Zadjust=0.35, Radjust=2.05,\
#                       WithWallCAD=False, SpecBins=10, PixelSamples=250)

# tok.synth_camera_test(CameraType=1, TorAngleDeg=-135, Title="45_slit?",\
#                       IndLightSize=0.001, BoundingCylMult=1.5, ZoomMult=1.695,\
#                       Zadjust=0.35, Radjust=0.0,\
#                       WithWallCAD=False, SpecBins=10, PixelSamples=100)

tok.synth_camera_test(CameraType=1, TorAngleDeg=-135, Title="64_tallslitcheck",\
                    IndLightSize=0.001, BoundingCylMult=1.5, ZoomMult=1.68,\
                    Zadjust=0.33, Radjust=0.0,\
                    WithWallCAD=False, SpecBins=10, PixelSamples=250,\
                    major_radius = 2.96, minor_radius = 1.25)