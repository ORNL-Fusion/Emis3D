"""
tests/test_fluxshell_library.py

FluxShell in a radDist library config. A flux shell is centred on the
magnetic axis whatever start position it is given, but make_raddists built
one per start position of the grid: the identical radDist, ray-traced again
and again and written to the same file (one shape on a 10 x 20 grid: 126
builds for one radDist). It is now built once per shape, and needs no grid.
"""

import pytest
import yaml

from emis3d import mock
from emis3d.pipeline import make_raddists
from emis3d.raddist_library import RadDistLibrary

BASE = {"tokamakName": "MOCK", "shot": 1, "folder": "lib", "injectionLocation": 225.0, "units": "Brightness", "eqFileName": mock.EQ_FILE,
        "workers": 1, "rays": {"pixelSamples": 5, "seed": 0}}
GRID = {"rLimits": [], "zLimits": [], "NumR": 6, "NumZ": 8}
SHELL = {"distType": "FluxShell", "psi_center": [0.7, 0.9], "psi_width": [0.1]}
RING = {"distType": "ElongatedRing", "sigma_R": [0.2], "sigma_z": [0.2]}


@pytest.fixture
def ws_dir(tmp_path):
    (tmp_path / "emis3d.yaml").write_text("version: 1\n")
    (tmp_path / "raddists" / "MOCK").mkdir(parents=True)
    return tmp_path


def write(ws_dir, cfg):
    path = ws_dir / "raddists" / "MOCK" / "raddists.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_one_raddist_per_shape_whatever_the_grid(ws_dir):
    s = make_raddists(write(ws_dir, {**BASE, "grid": GRID, "types": {"shell": SHELL, "ring": RING}}), workspace=ws_dir, dry_run=True)
    shell, ring = (next(t for t in s["types"] if t["name"] == name) for name in ("shell", "ring"))
    assert (shell["shapes"], shell["start_positions"], shell["radDists"]) == (2, 1, 2)       # two shapes: two radDists
    assert ring["start_positions"] > 1 and ring["radDists"] == ring["start_positions"]        # a ring still sweeps the grid
    assert s["radDists"] == 2 + ring["radDists"]


def test_a_flux_shell_needs_no_grid(ws_dir):
    lib = RadDistLibrary.from_dict({**BASE, "types": {"shell": SHELL}})
    assert not lib.types[0].uses_start_positions
    assert make_raddists(write(ws_dir, {**BASE, "types": {"shell": SHELL}}), workspace=ws_dir, dry_run=True)["radDists"] == 2
    with pytest.raises(ValueError, match="no grid of start positions"):                       # the other types still do
        RadDistLibrary.from_dict({**BASE, "types": {"ring": RING}})


def test_the_plan_says_so(ws_dir, capsys):
    make_raddists(write(ws_dir, {**BASE, "grid": GRID, "types": {"shell": SHELL}}), workspace=ws_dir, dry_run=True, progress=True)
    out = capsys.readouterr().out
    assert "2 distinct shape(s), centred on the magnetic axis (no start positions) = 2 radDists" in out
    assert "Total: 2 radDists" in out


@pytest.mark.cherab
def test_each_shape_is_built_once(ws_dir):
    s = make_raddists(write(ws_dir, {**BASE, "grid": GRID, "types": {"shell": SHELL}}), workspace=ws_dir)
    files = sorted((ws_dir / "raddists" / "MOCK" / "lib" / "shell").rglob("*.json"))
    assert s["built"] == 2 and s["failed"] == [] and len(files) == 2
    assert {f.parent.name for f in files} == {"fluxShell_psi_0.7_width_0.1_asym_0_at_0", "fluxShell_psi_0.9_width_0.1_asym_0_at_0"}
