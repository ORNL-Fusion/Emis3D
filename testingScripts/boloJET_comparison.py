# boloJET_comparison.py
"""
This script compares the bolometer geometry created by the manual config files to those found
in github: https://github.com/cherab/jet/tree/master/cherab/jet/bolometry/kb5

NOTE: It appears that the GitHub version loads specific kb5v.obj files, which we do not have. Therefore,
the comparison may not be accurate.
"""


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np

from emis3d.Tokamak import Tokamak
from emis3d.Util import config_loader
import pandas as pd

t = Tokamak(
    tokamakName="JET",
    mode="Build",
    reflections=False,
    loadBolometers=True,
)


t_git = Tokamak(
    tokamakName="JET",
    mode="Analysis",
    reflections=False,
    loadBolometers=False,
)
# input: tokamakName, mode, reflections, eqFileName
t_git._load_config_file("JET", "Build", False, None)
info = config_loader("../tokamaks/JET/JET_KB5V_settings.yaml")
t_git.info["BOLOMETERS"] = info["BOLOMETERS"]  # type:ignore
t_git._tokamak_startup(loadBolometers=True)
t_git.bolometers[0].calc_etendues()
e_git = t_git.bolometers[0].etendues[:24]


# --- Read Ben's Etendue's
path = "../tokamaks/JET/KB5V Etendues.xlsx"
data = pd.read_excel(path, sheet_name="Sheet1")


# --- Print out information
ii = 0
foil = t.bolometers[ii].bolometer_camera[0]
slit = foil.slit
foil_git = t.bolometers[0].bolometer_camera[ii]
slit_git = foil_git.slit

print(f"\n----->{foil.name}<-----")
print(f"                   Ben Value|||   GitHub Value      ||| Difference")
print(
    f"Slit width:        {slit.dx:.4f}   |||\t {slit_git.dx:.4f} \t||| {abs(slit.dx - slit_git.dx):.4f}"
)
print(
    f"Slit height:       {slit.dy:.4f}   |||\t {slit_git.dy:.4f} \t||| {abs(slit.dy - slit_git.dy):.4f}"
)
print(
    f"Foil width:        {foil.x_width:.4f}   |||\t {foil_git.x_width:.4f} \t||| {abs(foil.x_width - foil_git.x_width):.4f}"
)
print(
    f"Foil height:       {foil.y_width:.4f}   |||\t {foil_git.y_width:.4f} \t||| {abs(foil.y_width - foil_git.y_width):.4f}"
)
print(
    f"Foil Position[0]: {foil.centre_point[0]:.3e} |||\t {foil_git.centre_point[0]:.3e} \t||| {abs(foil.centre_point[0] - foil_git.centre_point[0]):.3e}"
)
print(
    f"Foil Position[1]: {foil.centre_point[1]:.3e} |||\t {foil_git.centre_point[1]:.3e} \t||| {abs(foil.centre_point[1] - foil_git.centre_point[1]):.3e}"
)
print(
    f"Foil Position[2]: {foil.centre_point[2]:.3e} |||\t {foil_git.centre_point[2]:.3e} \t||| {abs(foil.centre_point[2] - foil_git.centre_point[2]):.3e}"
)
print(
    f"Slit Position[0]: {slit.centre_point[0]:.3e} |||\t {slit_git.centre_point[0]:.3e} \t||| {abs(slit.centre_point[0] - slit_git.centre_point[0]):.3e}"
)
print(
    f"Slit Position[1]: {slit.centre_point[1]:.3e} |||\t {slit_git.centre_point[1]:.3e} \t||| {abs(slit.centre_point[1] - slit_git.centre_point[1]):.3e}"
)
print(
    f"Slit Position[2]: {slit.centre_point[2]:.3e} |||\t {slit_git.centre_point[2]:.3e} \t||| {abs(slit.centre_point[2] - slit_git.centre_point[2]):.3e}"
)
print("-" * 60)


if True:
    f = plt.figure(figsize=(12, 8))
    f_ = f.add_subplot(1, 2, 1)
    ax1 = f.add_subplot(122)

    if t_git.info is not None:
        boloGroups = t_git.info["Bolometer Groups"]
        t_git._plot_bolometers(
            f_, boloGroupName=boloGroups[0], plot_chord_info=False, legend=True
        )

    # --- Plot each individual bolometer
    if t.info is not None:
        boloGroups = t.info["Bolometer Groups"]
        t._plot_first_wall(f_)
        t._plot_bolometers(
            f_, boloGroupName=boloGroups[0], plot_chord_info=False, legend=True
        )

    chan = np.arange(1, 25)
    ax1.plot(chan, data["Old Etendue"] * 1.0e7, label="Old Emis3D Etendue", marker="o")
    ax1.plot(
        chan, data["Factored Etendue"] * 1.0e7, label="Factored Etendue", marker="^"
    )
    ax1.plot(chan, np.abs(np.array(e_git)) * 1.0e7, label="GitHub Config", marker="s")
    ax1.set_xlabel("Channel Number")
    ax1.set_ylabel("Etendue (e-7 sr)")
    ax1.set_ylim(0, 2.5)
    ax1.legend()
    plt.tight_layout()
    plt.show()
