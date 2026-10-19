"""
tests/test_run_config.py

RunConfig validation, and loading a run config through Emis3D the way a
real run does (no ray tracing).
"""

import pytest
import yaml

from emis3d.Emis3D import Emis3D
from emis3d.run_config import RunConfig
from emis3d.workspace import Workspace

MINIMAL = {
    "tokamakName": "MOCK",
    "shot": 1,
    "dt": 0.2,
    "radDistDirectories_LocIndependent": ["ring"],
    "BOLOMETERS": {"POL_A": {"dataFileName": "1/POL_A.h5"}},
}


def test_defaults_are_filled_in():
    c = RunConfig.from_dict(dict(MINIMAL))
    assert c.scale_def == "linear" and c.error_definition == "exponential"
    assert c.helical_endpoint_weight == 1.0 and c.paths == {}


@pytest.mark.parametrize("change, match", [
    ({"scale_def": "liner"}, "scale_def"),
    ({"error_definition": "gaussian"}, "error_definition"),
    ({"dtt": 1.0}, "Unknown run-config keys"),
    ({"radDistDirectories_LocIndependent": []}, "No radDist directories"),
])
def test_bad_configs_are_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        RunConfig.from_dict({**MINIMAL, **change})


def test_error_definition_is_case_insensitive():
    assert RunConfig.from_dict({**MINIMAL, "error_definition": "CONSTANT"}).error_definition == "CONSTANT"


def test_missing_required_key():
    with pytest.raises(TypeError, match="shot"):
        RunConfig.from_dict({k: v for k, v in MINIMAL.items() if k != "shot"})


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "emis3d.yaml").write_text("version: 1\n")
    run = tmp_path / "runs" / "1" / "run.yaml"
    run.parent.mkdir(parents=True)
    run.write_text(yaml.safe_dump(MINIMAL))
    return tmp_path, run


def test_emis3d_loads_a_run_config_by_path(workspace, monkeypatch):
    ws_dir, run = workspace
    monkeypatch.delenv("EMIS3D_RUNS_DIR", raising=False)
    e = Emis3D(initialize=False)
    e._load_config_file(runConfigName=str(run))
    assert e.info["tokamakName"] == "MOCK" and e.info["shot"] == 1
    assert e.info["scale_def"] == "linear"                     # default filled in
    assert e.run_config_path == run.resolve()
    assert e.ws is not None and e.ws.fields["workspace"] == str(ws_dir.resolve())


def test_emis3d_loads_a_run_config_by_name(workspace):
    ws_dir, _ = workspace
    ws = Workspace.resolve(workspace=ws_dir, env={}, user_config=ws_dir / "none.yaml")
    e = Emis3D(initialize=False, workspace=ws)
    e._load_config_file(tokamakName="MOCK", runConfigName="1/run.yaml")
    assert e.info["shot"] == 1


def test_emis3d_explicit_path_argument(workspace):
    _, run = workspace
    e = Emis3D(initialize=False)
    e._load_config_file(pathFileName=run)
    assert e.info["tokamakName"] == "MOCK"


def test_invalid_config_names_the_file(workspace):
    _, run = workspace
    run.write_text(yaml.safe_dump({**MINIMAL, "scale_def": "liner"}))
    e = Emis3D(initialize=False)
    with pytest.raises(ValueError, match="Invalid run config .*run.yaml"):
        e._load_config_file(runConfigName=str(run))


def test_info_exists_before_loading():
    assert Emis3D(initialize=False).info == {}
