# emis3DTester.py
"""
Program w

TODO: Add each signal in each array for each time segment, see how that compares with TPF

WHERE I LEFT OFF:
0. Need to increase the number of toroidal revolutions for the helical distributions to at least 2x,
this is due to the exponential fitting of the toroidal distribution, some times one of the directions
does not decay to zero after one revolution, this introduces a discontinuty at the phi = mu location
which is not physical. The way to solve this would be to do at least two revolutions, so the exponential
distribution can decay to zero-ish (need to think about CQ implications, but that is normally elongated rings
anyways)
1. Need to finish the powerPerBin calculation, and calculation of the TPF
Currently the elongatedRing looks okay, but the helical is off. It looks like the code
is calculating each respective clockwise and counterClock contribution correctly, but when
you add them up the radiation is dis-jointed at the injection location (mu). Perhaps I need
to fit something along the lines that A1 + A2 = A?
2. Still need to incorporate wall detectors into the code
3. Edit the code so it can run for more than one time step (easy-ish)
4. Compare incident radiation on the wall detectors to what is observed experimentally
5.


The new fitting routine is not working that well.... it seems to drop off too quickly
TODO: TODO: TODO: Increase the number of revolutions for the helical distribtion to 2x in both directions,
this should help with the fitting and ensure a continous TPF distribution.

NOW: The von-mises fitting does not go to zero fast enough, so the TPF distribution is discontinous
, put some limitiation of the endpoints of the distribution

Another bug: clockwise rev1 is higher than clockwise rev0!!!

"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np

import main.Emis3D as Emis3D
import scipy.integrate
import multiprocessing as mp
import main.Util_emis3D as Util_emis3D

evalTimes = np.arange(2119, 2127, 0.3)

phi = []
powerPerBin = []
chisq = []
pf = []
mf = []
dis = []
tpf = []
t = None


evalTime = 2122.0

t = Emis3D.Emis3D(
    verbose=True,
    initialize=False,
)
t._load_config_file(
    tokamakName="DIII-D",
    runConfigName="184407/184407_runConfig.yaml",
)
# t._load_bolometer_data()

t._load_bestFits(
    path="/Users/plh/Documents/git/Emis3D_Refactor/inputs/DIII-D/runs/184407/184407_bestFits_2122.00.h5"
)
t._plot_bestFit(evalTime=float(2122.0), save=True)
