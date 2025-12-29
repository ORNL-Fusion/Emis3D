# fieldLineTester.py
"""
Scripts takes a input R, z array and calculates the field line around the vessel
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from numpy import deg2rad

from main.Tokamak import Tokamak

t = Tokamak(
    tokamakName="DIII-D",
    mode="Analysis",
    reflections=False,
    eqFileName="g184407.02100",
    loadBolometers=True,
)


R0 = [2.1584]
z0 = [0.601]
# R0 = [2.1]
# z0 = [0.1]
startPhi = 225
t.set_fieldlines(startR=R0, startZ=z0, startPhi=deg2rad(startPhi), numTransists=2.0)

t.plot(fieldLineStartPhi=startPhi)
