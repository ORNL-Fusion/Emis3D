# sxrChordTester.py
"""
Loads all of the bolometers on the diagnostic, calculates
the chord position based off the SXR input file(s) and overlays
the measured chords in the SXR input file(s).
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np

from main.Tokamak import Tokamak

t = Tokamak(
    tokamakName="SPARC_FDR",
    mode="Build",
    reflections=False,
    loadBolometers=True,
)

plt.ion()

"""
# Plot in 3D to see if it is at the correct toroidal location
if True:
    t.plot()
"""
# --- Plot each individual bolometer
if t.info is not None:
    boloGroups = t.info["Bolometer Groups"]
    bolometers = t.bolometers

    num_figs = len(boloGroups)
    num_rows = int(np.ceil(num_figs / 4))  # no more than 4 across
    f = plt.figure(figsize=(15, 8))
    initilized = False

    # --- Loop over each bolometer group
    for ii, boloGroup in enumerate(boloGroups):
        f_ = f.add_subplot(num_rows, int(num_figs / num_rows), ii + 1)
        t._plot_first_wall(f_)
        t._plot_bolometers(
            f_,
            boloGroupName=boloGroup,
            plot_chord_info=True,
            plot_etendue=["SX45F07"],
            legend=True,
        )

    plt.tight_layout()
    plt.show()
