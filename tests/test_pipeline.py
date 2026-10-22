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

RING = {"distType": "ElongatedRing", "tokamakName": "MOCK", "saveRunsDirectoryName": "lib", "units": "Brightness",
        "injectionLocation": 225.0, "eqFileName": None, "numProcessors": 1,
        "sigma_R_vals": [0.15, 0.2], "sigma_z_vals": [0.2],
        "GRID": {"rLimits": [], "zLimits": [], "NumRStartGrid": 4, "NumZStartGrid": 4},
        "BOLOMETER_PROPS": {"pixelSamples": 5, "seed": 0}}


@pytest.fixture
def ws_dir(tmp_path):
    (tmp_path / "emis3d.yaml").write_text("version: 1\n")
    (tmp_path / "raddists" / "MOCK").mkdir(parents=True)
    return tmp_path


def write_cfg(ws_dir, cfg, name="ring.yaml"):
    path = ws_dir / "raddists" / "MOCK" / name
    path.write_text(yaml.safe_dump(cfg))
    return path


# ---------------------------------------------------------------------------
# make_raddists
# ---------------------------------------------------------------------------

def test_dry_run_counts_shapes_and_positions(ws_dir):
    s = make_raddists(write_cfg(ws_dir, RING), workspace=ws_dir, dry_run=True)
    assert s["distType"] == "ElongatedRing" and s["shapes"] == 2 and s["start_positions"] > 0
    assert s["radDists"] == s["shapes"] * s["start_positions"]
    assert s["folder"].endswith("raddists/MOCK/lib")


def test_config_by_name_and_tokamak_from_the_config(ws_dir):
    write_cfg(ws_dir, RING)
    assert make_raddists("ring.yaml", tokamak="MOCK", workspace=ws_dir, dry_run=True)["shapes"] == 2


@pytest.mark.parametrize("dist, keys", [
    ("Helical", {"sigma_R_vals": [0.05], "sigmaKernel": 0.02, "numFieldLines": 10, "eqFileName": "MOCK_solovev.geqdsk"}),
    ("HelicalRing", {"sigma_R_vals": [0.05], "sigma_z_vals": [0.05], "eqFileName": "MOCK_solovev.geqdsk"}),
    ("ElongatedRing", {"sigma_R_vals": [0.15], "sigma_z_vals": [0.2]}),          # no rotationAngles
])
def test_only_the_keys_a_type_uses_are_required(ws_dir, dist, keys):
    cfg = {k: v for k, v in RING.items() if k not in ("sigma_R_vals", "sigma_z_vals")}
    cfg.update(distType=dist, **keys)
    assert make_raddists(write_cfg(ws_dir, cfg), workspace=ws_dir, dry_run=True)["shapes"] == 1


@pytest.mark.parametrize("change, match", [
    ({"sigma_z_vals": None}, "missing"),
    ({"distType": "Banana"}, "Unknown distType"),
    ({"GRID": None}, "missing"),
])
def test_bad_configs_are_reported(ws_dir, change, match):
    cfg = {**RING, **change}
    cfg = {k: v for k, v in cfg.items() if v is not None}
    with pytest.raises(ValueError, match=match):
        make_raddists(write_cfg(ws_dir, cfg), workspace=ws_dir, dry_run=True)


def test_missing_config_is_reported(ws_dir):
    with pytest.raises(FileNotFoundError):
        make_raddists("nope.yaml", tokamak="MOCK", workspace=ws_dir)


def test_make_raddists_command_dry_run(ws_dir, capsys):
    assert main(["make-raddists", str(write_cfg(ws_dir, RING)), "--workspace", str(ws_dir), "--dry-run"]) == 0
    assert "Would build" in capsys.readouterr().out


@pytest.mark.cherab
def test_make_raddists_builds_files(ws_dir):
    cfg = {**RING, "sigma_R_vals": [0.2], "GRID": {"rLimits": [1.7, 1.9], "zLimits": [0.0, 0.0], "NumRStartGrid": 2, "NumZStartGrid": 1}}
    s = make_raddists(write_cfg(ws_dir, cfg), workspace=ws_dir)
    files = sorted((ws_dir / "raddists" / "MOCK" / "lib").rglob("*.json"))
    assert len(files) == s["radDists"] == 2


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
