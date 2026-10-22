"""
tests/test_raddist_library.py

The radDist library config (emis3d.raddist_library): one file for every
radDist type of a shot, checked when it is loaded and expanded into the
settings each radDist is built with.
"""

import pytest
import yaml

from emis3d.raddist_library import RadDistLibrary

BASE = {
    "tokamakName": "MOCK", "shot": 7, "injectionLocation": 225, "units": "Brightness",
    "eqFileName": "MOCK_solovev.geqdsk", "numBins": 36, "workers": 2,
    "rays": {"pixelSamples": 200, "seed": 1, "integrationStep": 0.02},
    "grid": {"rLimits": [], "zLimits": [], "NumR": 4, "NumZ": 6},
}


def lib(types, **top):
    return RadDistLibrary.from_dict({**BASE, **top, "types": types})


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------

def test_shared_settings_reach_every_radDist():
    L = lib({"ring": {"distType": "ElongatedRing", "sigma_R": 0.2, "sigma_z": 0.3}})
    (t,) = L.types
    cfg = L.radDist_config(t, L.shape_sets(t)[0])
    assert cfg == {
        "distType": "ElongatedRing", "tokamakName": "MOCK", "shotNumber": 7, "units": "Brightness",
        "injectionLocation": 225.0, "numBins": 36, "eqFileName": "MOCK_solovev.geqdsk", "integrationStep": 0.02,
        "BOLOMETER_PROPS": {"pixelSamples": 200, "seed": 1}, "saveRunsDirectoryName": "7/ring",   # folder defaults to shot
        "sigma_R": 0.2, "sigma_z": 0.3, "rotationAngle": 0.0,
    }


def test_a_type_can_override_shared_settings():
    L = lib({"helical": {"distType": "Helical", "sigma_R": 0.05, "sigmaKernel": 0.01, "numFieldLines": 50,
                         "grid": {"rLimits": [1.9, 2.0], "zLimits": [0, 0], "NumR": 2, "NumZ": 1},
                         "rays": {"pixelSamples": 50}, "injectionLocation": 30}})
    (t,) = L.types
    cfg = L.radDist_config(t, L.shape_sets(t)[0])
    assert t.settings["grid"]["NumR"] == 2 and cfg["injectionLocation"] == 30.0
    assert cfg["BOLOMETER_PROPS"] == {"pixelSamples": 50, "seed": 0}   # rays replaced as a whole: defaults fill the rest


def test_lists_are_swept_and_tidy_names_are_translated():
    L = lib({"helical": {"distType": "Helical", "sigma_R": [0.03, 0.05], "sigmaKernel": [0.01, 0.02],
                         "numFieldLines": 100, "numTransits": 2}})
    shapes = L.shape_sets(L.types[0])
    assert len(shapes) == 4
    assert all(s["numTransists"] == 2 and "numTransits" not in s for s in shapes)   # the classes' own name
    assert {(s["sigma_R"], s["sigmaKernel"]) for s in shapes} == {(0.03, 0.01), (0.03, 0.02), (0.05, 0.01), (0.05, 0.02)}


def test_duplicate_rings_are_built_once():
    L = lib({"ring": {"distType": "ElongatedRing", "sigma_R": [0.2, 0.3], "sigma_z": [0.2, 0.3], "rotationAngle": [0, 90]}})
    shapes = L.shape_sets(L.types[0])
    # circles need one angle, and (0.2, 0.3, 90) is (0.3, 0.2, 0): 8 combinations -> 4 shapes
    assert len(shapes) == 4


def test_simulation_files_resolve_next_to_the_config(tmp_path):
    path = tmp_path / "raddists.yaml"
    path.write_text(yaml.safe_dump({**BASE, "types": {"nim": {"distType": "NIMROD", "files": ["a.json", "/abs/b.json"], "timesteps": [1, 2]}}}))
    L = RadDistLibrary.load(path)
    shapes = L.shape_sets(L.types[0])
    assert [(s["sourceFile"], s["timestep"]) for s in shapes] == [
        (str(tmp_path / "a.json"), 1), (str(tmp_path / "a.json"), 2), ("/abs/b.json", 1), ("/abs/b.json", 2)]


def test_select_by_name_and_enabled():
    L = lib({"ring": {"distType": "ElongatedRing", "sigma_R": 0.2, "sigma_z": 0.2},
             "tube": {"distType": "SquareTube", "dR": 0.1, "dz": 0.1, "enabled": False}})
    assert [t.name for t in L.select()] == ["ring"]
    assert [t.name for t in L.select(["tube"])] == ["tube"]
    assert L.folder_for(L.types[0]) == "7/ring"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("types, top, match", [
    ({"r": {"distType": "ElongatedRing", "sigma_R": 0.2, "sigma_z": 0.2, "sigma_R_vals": [0.2]}}, {}, r"types.r: unknown key\(s\) \['sigma_R_vals'\]"),
    ({"r": {"distType": "ElongatedRing", "sigma_R": 0.2}}, {}, r"types.r: ElongatedRing needs \['sigma_z'\]"),
    ({"r": {"distType": "Banana"}}, {}, "distType must be one of"),
    ({"h": {"distType": "Helical", "sigma_R": 0.05, "sigmaKernel": 0.01, "numFieldLines": 5}}, {"eqFileName": None}, "needs an eqFileName"),
    ({"r": {"distType": "ElongatedRing", "sigma_R": [], "sigma_z": 0.2}}, {}, "sigma_R is an empty list"),
    ({"r": {"distType": "ElongatedRing", "sigma_R": 0.2, "sigma_z": 0.2, "rays": {"pixel_samples": 5}}}, {}, r"rays: unknown key"),
    ({"r": {"distType": "ElongatedRing", "sigma_R": 0.2, "sigma_z": 0.2}}, {"GRID": {}}, "unknown setting"),
    ({"r": {"distType": "ElongatedRing", "sigma_R": 0.2, "sigma_z": 0.2}}, {"units": "Watts"}, "units must be one of"),
    ({}, {}, "at least one type"),
])
def test_problems_are_named(types, top, match):
    with pytest.raises(ValueError, match=match):
        lib(types, **top)


def test_old_format_is_refused():
    old = {"distType": "ElongatedRing", "tokamakName": "MOCK", "injectionLocation": 225, "sigma_R_vals": [0.2],
           "sigma_z_vals": [0.2], "rotationAngles": [0], "GRID": {"NumRStartGrid": 2}, "numProcessors": 2}
    with pytest.raises(ValueError, match="unknown setting"):
        RadDistLibrary.from_dict(old)


def test_every_problem_is_reported_at_once():
    with pytest.raises(ValueError) as e:
        lib({"a": {"distType": "ElongatedRing", "sigma_R": 0.2}, "b": {"distType": "SquareTube", "dR": 0.1, "color": "red"}})
    assert "types.a" in str(e.value) and "types.b" in str(e.value)


def test_the_shipped_example_loads():
    from pathlib import Path

    L = RadDistLibrary.load(Path(__file__).parents[1] / "examples" / "radDist" / "raddists_example.yaml")
    assert [t.name for t in L.select()] == ["elongatedRing", "helical"]
