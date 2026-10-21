"""
tests/test_raddist_save.py

Review issue 15: every radDist parameter that changes the shape is part of the
save path, so different radDists are never written to the same file, and
make_radDists builds each distinct shape once.
"""

import json

import numpy as np
import pytest

from emis3d.radDist import M3DC1, NIMROD, ElongatedRing, Helical, HelicalRing, SquareTube
from emis3d.Util_radDist import shape_parameter_sets
from emis3d.workspace import Workspace, set_workspace


def bare(cls, **info):
    """An instance with just the info needed for naming (no tokamak, no ray tracing)."""
    obj = cls.__new__(cls)
    obj.info = dict(info)
    obj.data = {}
    return obj


RING = dict(distType="elongatedRing", sigma_R=0.3, sigma_z=0.5, rotationAngle=30.0, startR=1.75, startZ=0.15)
HELICAL = dict(distType="Helical", sigma_R=0.2, sigmaKernel=0.05, numTransists=1.0, numFieldLines=200,
               startPhi=225.0, startR=1.75, startZ=0.15)
HRING = dict(distType="HelicalRing", sigma_R=0.2, sigma_z=0.3, numTransists=1.0, startPhi=225.0, startR=1.75, startZ=0.15)
TUBE = dict(distType="SquareTube", dR=0.1, dz=0.2, startR=1.75, startZ=0.15)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls, info, folder", [
    (ElongatedRing, RING, "elongatedRing_sigma_R_0.3_sigma_z_0.5_rot_30"),
    (Helical, HELICAL, "Helical_sigma_R_0.2_kernel_0.05_transits_1_lines_200_phi_225"),
    (HelicalRing, HRING, "HelicalRing_sigma_R_0.2_sigma_z_0.3_transits_1_phi_225"),
    (SquareTube, TUBE, "SquareTube_dR_0.1_dz_0.2"),
])
def test_folder_and_file_names(cls, info, folder):
    assert bare(cls, **info)._save_names() == (folder, "R_1.7500_z_0.1500.json")


@pytest.mark.parametrize("cls, info, key, other", [
    (ElongatedRing, RING, "rotationAngle", 60.0),      # used to overwrite each other
    (ElongatedRing, RING, "sigma_z", 0.6),
    (Helical, HELICAL, "sigmaKernel", 0.08),            # used to overwrite each other
    (Helical, HELICAL, "startPhi", 45.0),
    (Helical, HELICAL, "numFieldLines", 400),
    (HelicalRing, HRING, "numTransists", 2.0),
    (SquareTube, TUBE, "dz", 0.3),
])
def test_every_shape_parameter_changes_the_path(cls, info, key, other):
    a = bare(cls, **info)._save_names()
    b = bare(cls, **{**info, key: other})._save_names()
    assert a != b


def test_start_positions_closer_than_a_centimetre_get_different_files():
    """Two decimals used to give R_1.75 for both."""
    a = bare(ElongatedRing, **RING)._save_names()[1]
    b = bare(ElongatedRing, **{**RING, "startR": 1.7512})._save_names()[1]
    assert a != b and b == "R_1.7512_z_0.1500.json"


def test_missing_optional_parameter_is_left_out():
    info = {k: v for k, v in RING.items() if k != "rotationAngle"}
    assert bare(ElongatedRing, **info)._save_names()[0] == "elongatedRing_sigma_R_0.3_sigma_z_0.5"


@pytest.mark.parametrize("cls", [NIMROD, M3DC1])
def test_simulation_names_use_file_and_timestep(cls):
    obj = bare(cls, distType=cls.NAME, sourceFile="/data/sims/shot123_prad.json", timestep=40)
    assert obj._save_names() == (f"{cls.NAME}_shot123_prad", "t_40.json")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def test_save_writes_atomically_into_the_workspace(tmp_path):
    set_workspace(Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml"))
    try:
        for angle in (30.0, 60.0):
            obj = bare(ElongatedRing, **{**RING, "rotationAngle": angle},
                       tokamakName="MOCK", saveRunsDirectoryName="lib")
            obj.data = {"Brightness": {"x": np.arange(3.0)}}
            obj.saveRadDist()
    finally:
        set_workspace(None)
    lib = tmp_path / "raddists" / "MOCK" / "lib"
    files = sorted(p.relative_to(lib).as_posix() for p in lib.rglob("*") if p.is_file())
    assert files == [
        "elongatedRing_sigma_R_0.3_sigma_z_0.5_rot_30/R_1.7500_z_0.1500.json",
        "elongatedRing_sigma_R_0.3_sigma_z_0.5_rot_60/R_1.7500_z_0.1500.json",
    ]                                                            # no temporary files left behind
    saved = json.loads((lib / files[1]).read_text())
    assert saved["info"]["rotationAngle"] == 60.0 and saved["data"]["Brightness"]["x"] == [0.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# Distinct shapes for make_radDists
# ---------------------------------------------------------------------------

def triples(sets):
    return [(s["sigma_R"], s["sigma_z"], s["rotationAngle"]) for s in sets]


def test_rotated_twins_and_circles_are_built_once():
    sets = shape_parameter_sets("ElongatedRing", [0.3, 0.5], [0.3, 0.5], [0.0, 90.0])
    assert triples(sets) == [(0.3, 0.3, 0.0), (0.3, 0.5, 0.0), (0.5, 0.3, 0.0), (0.5, 0.5, 0.0)]


def test_unique_ninety_degree_ellipse_is_kept():
    """The old rule dropped (0.5, 0.3, 90) because 0.3 is a sigma_R value, but its
    twin (0.3, 0.5, 0) was never requested, so it is a unique shape."""
    sets = shape_parameter_sets("ElongatedRing", [0.3, 0.5], [0.3], [0.0, 90.0])
    assert triples(sets) == [(0.3, 0.3, 0.0), (0.5, 0.3, 0.0), (0.5, 0.3, 90.0)]


def test_angles_half_a_turn_apart_are_the_same_ellipse():
    sets = shape_parameter_sets("ElongatedRing", [0.3], [0.5], [30.0, 210.0, 60.0])
    assert triples(sets) == [(0.3, 0.5, 30.0), (0.3, 0.5, 60.0)]


def test_types_that_ignore_parameters_are_not_repeated():
    assert shape_parameter_sets("Helical", [0.2, 0.3], [0.1, 0.2, 0.3], [0.0, 30.0]) == [
        {"sigma_R": 0.2}, {"sigma_R": 0.3}]
    assert len(shape_parameter_sets("HelicalRing", [0.2, 0.3], [0.1, 0.2], [0.0, 30.0, 60.0])) == 4
    assert shape_parameter_sets("SquareTube", [0.2, 0.3], [0.1], [0.0]) == [{}]
