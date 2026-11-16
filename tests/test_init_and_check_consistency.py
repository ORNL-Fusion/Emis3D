"""
tests/test_init_and_check_consistency.py

Two places where the tools disagreed with a run:

* the example run.yaml that `emis3d init` writes was not a valid run config
  (empty radDist folders with the library solver), so `emis3d check` and
  `emis3d run` rejected it before looking at any file;
* `emis3d check` compared location-dependent radDists with the run config's
  injectionLocation even when the run named an injector:, whose angle is
  what Emis3D uses. With injector: and no injectionLocation (0 by default)
  it reported every correct radDist as wrong, and passed wrong ones made
  for 0 deg.
"""

import json

import pytest
import yaml

from emis3d import cli
from emis3d.pipeline import check
from emis3d.run_config import RunConfig
from test_calibration import write_workspace


# ---------------------------------------------------------------------------
# emis3d init
# ---------------------------------------------------------------------------

def test_the_example_run_config_is_valid():
    cfg = RunConfig.from_dict(yaml.safe_load(cli.RUN_TEMPLATE))
    assert cfg.solver == "library" and cfg.radDistDirectories_LocIndependent and cfg.eval_times() == [10.0]


def test_check_reads_the_example_and_names_the_missing_files(tmp_path, monkeypatch):
    monkeypatch.delenv("EMIS3D_RUNS_DIR", raising=False)
    target = tmp_path / "ws"
    assert cli.main(["init", str(target)]) == 0
    found = check(target / "runs" / "example" / "run.yaml")
    assert found[0].level == "ok" and "run config is valid" in found[0].message
    errors = " | ".join(f.message for f in found if f.level == "error")
    assert "POL_A: data file not found" in errors and "radDist folder not found" in errors   # what is still to be provided


# ---------------------------------------------------------------------------
# emis3d check with injector:
# ---------------------------------------------------------------------------

def set_raddist_injection(root, degrees):
    for f in (root / "raddists" / "MOCK" / "ring").glob("*.json"):
        saved = json.loads(f.read_text())
        saved["info"]["injectionLocation"] = degrees
        f.write_text(json.dumps(saved))


def injection_errors(run):
    return [f.message for f in check(run) if f.level == "error" and "injectionLocation" in f.message]


def test_raddists_at_the_injectors_angle_pass(tmp_path):
    run = write_workspace(tmp_path, tstart=10.0, loc_dependent=True, injector="SPI225")   # MOCK's SPI225 is at 225 deg
    set_raddist_injection(tmp_path, 225.0)
    assert injection_errors(run) == []


def test_raddists_at_another_angle_are_reported_against_the_injector(tmp_path):
    run = write_workspace(tmp_path, tstart=10.0, loc_dependent=True, injector="SPI225")
    msgs = injection_errors(run)                                                # the fabricated radDists are for 0 deg
    assert len(msgs) == 3 and all("differs from injector SPI225's 225" in m for m in msgs)


@pytest.mark.parametrize("raddist_angle, ok", [(225.0, True), (-135.0, True), (585.0, True), (45.0, False)])
def test_without_an_injector_the_run_configs_angle_is_used(tmp_path, raddist_angle, ok):
    run = write_workspace(tmp_path, tstart=10.0, loc_dependent=True, injectionLocation=225.0)
    set_raddist_injection(tmp_path, raddist_angle)
    msgs = injection_errors(run)
    assert (msgs == []) if ok else (len(msgs) == 3 and "differs from the run config's 225.0" in msgs[0])
