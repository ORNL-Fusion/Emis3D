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


tokamakName = "DIII-D"

tok = Tokamak(
    tokamakName="DIII-D",
    mode="Analysis",
    reflections=False,
    loadBolometers=True,
    )

#breakpoint()

# tok.synth_camera_test(CameraType=0, TorAngleDeg=135, Title="gettingthere10",\
#                     WithWallCAD=True, SpecBins=10, PixelSamples=250,\
#                     ZoomMult=1.0)

tok.synth_camera_test(CameraType=1, TorAngleDeg=135, Title="gettingthere10",\
                    WithWallCAD=True, SpecBins=10, PixelSamples=250,\
                    ZoomMult=1.35, Zadjust=0.75)

# tok.synth_camera_test(CameraType=0, TorAngleDeg=-135, Title="toroidal-135",\
#                       SpecBins=10, PixelSamples=250)
