"""
tests/test_camera_config_errors.py

A camera whose configuration cannot be loaded stops the tokamak from being
built, with a message that names the file and what is wrong. It used to
print a line and carry on with a camera that had no geometry, which failed
later, somewhere else, with an AttributeError ('Bolometer' object has no
attribute 'bolometer_camera').
"""

import pytest
import yaml

from emis3d.mock import CameraSpec, mock_workspace
from emis3d.pipeline import check
from emis3d.tokamak import Tokamak
from test_calibration import write_workspace


def own_machine(path):
    """
    A workspace with its own MOCK machine, to be damaged. Its cameras have
    names the built-in MOCK does not, so a file removed here is not found
    among the built-in machine's files instead.
    """
    return mock_workspace(path, cameras=[CameraSpec("CAM_A", 90.0), CameraSpec("CAM_B", 225.0)])


@pytest.fixture
def ws(tmp_path):
    return own_machine(tmp_path)


def camera_file(ws, name="CAM_B"):
    return ws.machine_file("MOCK", "inputs", "sxrInfo", f"{name}_config.yaml")


def edit(path, **changes):
    cfg = yaml.safe_load(path.read_text())
    for key, value in changes.items():
        if value is None:
            cfg.pop(key)
        else:
            cfg[key] = value
    path.write_text(yaml.safe_dump(cfg))


def build(ws):
    return Tokamak(tokamakName="MOCK", mode="Analysis", loadBolometers=True, workspace=ws)


def test_the_untouched_machine_loads(ws):
    assert [b.name for b in build(ws).bolometers] == ["CAM_A", "CAM_B"]


def test_a_missing_camera_file_is_named(ws):
    camera_file(ws).unlink()
    with pytest.raises(FileNotFoundError) as err:
        build(ws)
    assert "CAM_B_config.yaml" in str(err.value) and "not found for MOCK" in str(err.value) and "machines" in str(err.value)


@pytest.mark.parametrize("key", ["NAME", "GROUP_NAME", "BUILD_TYPE"])
def test_a_missing_setting_is_named(ws, key):
    edit(camera_file(ws), **{key: None})
    with pytest.raises(ValueError, match=rf"CAM_B_config.yaml lacks \['{key}'\]"):
        build(ws)


def test_an_unsupported_build_type_is_an_error(ws):
    edit(camera_file(ws), BUILD_TYPE="FROM CAD")
    with pytest.raises(ValueError, match="BUILD_TYPE 'FROM CAD' is not supported"):
        build(ws)


def test_an_empty_camera_file_is_an_error(ws):
    camera_file(ws).write_text("# nothing here\n")
    with pytest.raises(ValueError, match="empty or is not a mapping"):
        build(ws)


def test_tokamak_without_cameras_does_not_read_them(ws):
    camera_file(ws).unlink()
    Tokamak(tokamakName="MOCK", mode="Analysis", loadBolometers=False, workspace=ws)   # e.g. field-line work: no cameras needed


def test_check_reports_it_as_a_finding(tmp_path):
    run = write_workspace(tmp_path, tstart=10.0)
    ws = own_machine(tmp_path)                                                  # the workspace's own MOCK, found first
    camera_file(ws).unlink()
    found = check(run)
    errors = [f.message for f in found if f.level == "error"]
    assert any("machine MOCK" in m and "CAM_B_config.yaml" in m and "not found" in m for m in errors)
