# sxrChordTester.py
"""
Loads all of the bolometers on the diagnostic, calculates
the chord position based off the SXR input file(s) and overlays
the measured chords in the SXR input file(s).

TODO:
1. Some weird bug happens when I load more than one bolometer, it's like they don't
trace the rays to the wall correctly

"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np
from raysect.core import Point2D

from main.Tokamak import Tokamak


def point3d_to_rz(point):
    return Point2D(np.hypot(point.x, point.y), point.z)


t = Tokamak(
    tokamakName="DIII-D",
    mode="Build",
    reflections=False,
    eqFileName="g184407.02100",
    loadBolometers=True,
)


# Plot in 3D to see if it is at the correct toroidal location
if True:
    t.plot()

# --- Plot each individual bolometer
if True:
    num_figs = len(t.bolometers)
    num_rows = int(np.ceil(num_figs / 4))  # no more than 4 across
    f = plt.figure(figsize=(15, 8))
    # f_ = f.add_subplot(111)
    for ii, bolo in enumerate(t.bolometers):

        f_ = f.add_subplot(num_rows, int(num_figs / num_rows), ii + 1)

        t._plot_first_wall(f_)
        if "r0" in bolo.info:
            r0, z0, rf, zf = (
                bolo.info["r0"],
                bolo.info["z0"],
                bolo.info["rf"],
                bolo.info["zf"],
            )
            f_.plot([r0, rf], [z0, zf], color="green")
        for foil in bolo.bolometer_camera:
            slit_centre = foil.slit.centre_point
            slit_centre_rz = point3d_to_rz(slit_centre)
            f_.plot(slit_centre_rz[0], slit_centre_rz[1], "ko")
            origin, hit, _ = foil.trace_sightline()
            centre_rz = point3d_to_rz(foil.centre_point)
            f_.plot(centre_rz[0], centre_rz[1], "kx")
            origin_rz = point3d_to_rz(origin)
            hit_rz = point3d_to_rz(hit)
            f_.plot([origin_rz[0], hit_rz[0]], [origin_rz[1], hit_rz[1]], "k")
            f_.text(hit_rz[0], hit_rz[1], foil.name)
        f_.set_title(bolo.name)

    plt.tight_layout()
    plt.show()
