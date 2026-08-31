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
from itertools import batched
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
    MAX_PER_PLOT = 4
    boloGroups = [
        list(batch) for batch in batched(t.info["Bolometer Groups"], MAX_PER_PLOT)
    ]
    NUM_PLOTS = len(boloGroups)

    bolometers = t.bolometers

    for boloGroup in boloGroups:
        f = plt.figure(figsize=(15, 8))
        initilized = False

        # --- Loop over each bolometer group
        for ii, boloGroupName in enumerate(boloGroup):
            f_ = f.add_subplot(1, 4, ii + 1)
            t._plot_first_wall(f_)
            t._plot_bolometers(
                f_,
                boloGroupName=boloGroupName,
                plot_chord_info=True,
                plot_etendue=[],
                legend=True,
            )

        plt.tight_layout()
    plt.show()
