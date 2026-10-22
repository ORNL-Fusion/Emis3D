"""
tests/test_pipeline.py

emis3d.pipeline (make_raddists, run, check) and the `emis3d make-raddists`,
`emis3d check` and `emis3d run` commands, on the MOCK machine.
"""

import json

import h5py
import pytest
import yaml

from emis3d.cli import main
from emis3d.pipeline import check, make_raddists
from test_calibration import write_workspace

LIBRARY = {
    "tokamakName": "MOCK", "shot": 1, "folder": "lib", "injectionLocation": 225.0, "units": "Brightness",
    "workers": 1, "rays": {"pixelSamples": 5, "seed": 0},
    "grid": {"rLimits": [], "zLimits": [], "NumR": 4, "NumZ": 4},
    "types": {"ring": {"distType": "ElongatedRing", "sigma_R": [0.15, 0.2], "sigma_z": [0.2]}},
}


@pytest.fixture
def ws_dir(tmp_path):
    (tmp_path / "emis3d.yaml").write_text("version: 1\n")
    (tmp_path / "raddists" / "MOCK").mkdir(parents=True)
    return tmp_path


def write_cfg(ws_dir, cfg, name="raddists.yaml"):
    path = ws_dir / "raddists" / "MOCK" / name
    path.write_text(yaml.safe_dump(cfg))
    return path


# ---------------------------------------------------------------------------
# make_raddists
# ---------------------------------------------------------------------------

def test_dry_run_counts_each_type(ws_dir):
    s = make_raddists(write_cfg(ws_dir, LIBRARY), workspace=ws_dir, dry_run=True)
    (ring,) = s["types"]
    assert ring["name"] == "ring" and ring["shapes"] == 2 and ring["start_positions"] > 0
    assert s["radDists"] == ring["radDists"] == 2 * ring["start_positions"]
    assert ring["folder"].endswith("raddists/MOCK/lib/ring")


def test_config_by_name_with_tokamak(ws_dir):
    write_cfg(ws_dir, LIBRARY)
    assert make_raddists("raddists.yaml", tokamak="MOCK", workspace=ws_dir, dry_run=True)["radDists"] > 0
    with pytest.raises(ValueError, match="is for MOCK, not JET"):
        make_raddists(write_cfg(ws_dir, LIBRARY), tokamak="JET", workspace=ws_dir, dry_run=True)


def test_only_selects_types(ws_dir):
    cfg = {**LIBRARY, "types": {**LIBRARY["types"], "tube": {"distType": "SquareTube", "dR": 0.1, "dz": 0.1, "enabled": False}}}
    path = write_cfg(ws_dir, cfg)
    assert [t["name"] for t in make_raddists(path, workspace=ws_dir, dry_run=True)["types"]] == ["ring"]   # tube is off
    assert [t["name"] for t in make_raddists(path, workspace=ws_dir, only=["tube"], dry_run=True)["types"]] == ["tube"]
    with pytest.raises(ValueError, match="no type"):
        make_raddists(path, workspace=ws_dir, only=["nope"], dry_run=True)


def test_missing_config_is_reported(ws_dir):
    with pytest.raises(FileNotFoundError):
        make_raddists("nope.yaml", tokamak="MOCK", workspace=ws_dir)


def test_make_raddists_command_dry_run(ws_dir, capsys):
    assert main(["make-raddists", str(write_cfg(ws_dir, LIBRARY)), "--workspace", str(ws_dir), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "ring" in out and "Would build" in out


@pytest.mark.cherab
def test_make_raddists_builds_files_and_a_manifest(ws_dir, tmp_path):
    import json as _json

    import numpy as np

    r = np.linspace(1.05, 2.35, 14)
    z = np.linspace(-1.05, 1.05, 22)
    phi = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    Z, R, P = np.meshgrid(z, r, phi, indexing="ij")
    sim = ws_dir / "raddists" / "MOCK" / "sim.json"
    sim.write_text(_json.dumps({"pradArray": np.exp(-((R - 1.85) ** 2 + Z**2) / 0.1).tolist(), "r_list": r.tolist(),
                                "z_list": z.tolist(), "phi_list": phi.tolist()}))
    cfg = {**LIBRARY, "grid": {"rLimits": [1.7, 1.9], "zLimits": [0.0, 0.0], "NumR": 2, "NumZ": 1},
           "types": {"ring": {"distType": "ElongatedRing", "sigma_R": 0.2, "sigma_z": 0.2},
                     "sim": {"distType": "NIMROD", "files": ["sim.json"], "timesteps": [3]}}}
    s = make_raddists(write_cfg(ws_dir, cfg), workspace=ws_dir)
    lib = ws_dir / "raddists" / "MOCK" / "lib"
    assert len(list((lib / "ring").rglob("*.json"))) == 2 and len(list((lib / "sim").rglob("t_3.json"))) == 1
    manifest = yaml.safe_load((lib / "manifest.yaml").read_text())
    assert [t["name"] for t in manifest["built"]] == ["ring", "sim"] and manifest["library"]["types"]["sim"]["timesteps"] == [3]
    assert s["radDists"] == 3


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def levels(findings):
    return [f.level for f in findings]


def errors(findings):
    return [f.message for f in findings if f.level == "error"]


def test_a_good_run_config_passes(tmp_path):
    run = write_workspace(tmp_path, tstart=9.5, tend=10.5, Nt=3)
    found = check(run)
    assert errors(found) == [] and "warning" not in levels(found)
    assert any("3 evaluation time(s)" in f.message for f in found)


def test_check_finds_problems(tmp_path):
    run = write_workspace(tmp_path, tstart=12.0, BOLOMETERS={
        "POL_A": {"dataFileName": "1/POL_A.h5", "badChannels": ["NOPE"]},
        "POL_B": {"dataFileName": "1/missing.h5"},
        "GHOST": {"dataFileName": "1/POL_C.h5"},
    })
    msgs = " | ".join(errors(check(run)))
    assert "cameras not in the MOCK machine files: ['GHOST']" in msgs
    assert "POL_B: data file not found" in msgs
    assert "badChannels not in the data file: ['NOPE']" in msgs
    assert "outside the data" in msgs


def test_check_finds_raddist_problems(tmp_path):
    run = write_workspace(tmp_path)
    folder = tmp_path / "raddists" / "MOCK" / "ring"
    first = sorted(folder.glob("*.json"))[0]
    d = json.loads(first.read_text())
    d["info"]["radDistFormat"] = 1
    first.write_text(json.dumps(d))
    (tmp_path / "raddists" / "MOCK" / "empty").mkdir()
    cfg = yaml.safe_load(run.read_text())
    cfg["radDistDirectories_LocIndependent"] += ["empty", "absent"]
    run.write_text(yaml.safe_dump(cfg))
    msgs = " | ".join(errors(check(run)))
    assert "format 1, not 2: regenerate" in msgs
    assert "radDist folder is empty" in msgs and "radDist folder not found" in msgs


def test_check_reports_an_invalid_run_config(tmp_path):
    bad = tmp_path / "run.yaml"
    bad.write_text("shot: 1\nBOLOMETERS: {}\nnot_a_field: 3\n")
    assert "run config:" in errors(check(bad))[0]


def test_check_command_exit_codes(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    good = write_workspace(tmp_path / "a", tstart=10.0)
    assert main(["check", str(good)]) == 0
    bad = write_workspace(tmp_path / "b", tstart=99.0)
    assert main(["check", str(bad)]) == 1
    assert "error(s)" in capsys.readouterr().out


def test_check_finds_an_unreadable_data_file(tmp_path):
    run = write_workspace(tmp_path)
    with h5py.File(tmp_path / "measurements" / "MOCK" / "1" / "POL_A.h5", "a") as f:
        del f["TIME"]
    assert any("POL_A.h5 lacks ['TIME']" in m for m in errors(check(run)))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@pytest.mark.cherab
def test_run_command_end_to_end(tmp_path, capsys):
    from emis3d.mock import selftest

    selftest(workdir=tmp_path, pixel_samples=20, verbose=False)
    run = tmp_path / "runs" / "selftest" / "run.yaml"
    assert main(["check", str(run)]) == 0
    assert main(["run", str(run), "--times", "9.8", "10.0", "--no-plots"]) == 0
    out = capsys.readouterr().out
    assert "t = 9.8: TPF" in out and "t = 10: TPF" in out
    assert list((tmp_path / "runs" / "selftest" / "results").glob("*/*bestFits*"))
