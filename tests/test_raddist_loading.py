"""
tests/test_raddist_loading.py

Emis3D._load_radDists stops with a clear message when a radDist folder from
the run config is missing or yields nothing, instead of failing later in the
fit with "attempt to get argmin of an empty sequence".
"""

import json

import pytest

from emis3d.Emis3D import Emis3D
from emis3d.workspace import Workspace


def write_raddist(path):
    """A minimal radDist file with the fields loading needs."""
    data = {
        "Brightness": {"channelOrder": {"CAM": ["C1", "C2"]}, "ring": {"CAM": [[1.0], [2.0]]}},
        "Brightness_error": {"ring": {"CAM": [[0.0], [0.0]]}},
        "observed_phi_loc": {"ring": {"CAM": [[0.0], [0.0]]}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"info": {"units": "Brightness", "emissionNames": ["ring"]}, "data": data}))


@pytest.fixture
def run(tmp_path):
    ws = Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml")
    e = Emis3D(initialize=False, workspace=ws)
    e.info = {"tokamakName": "JET", "radDistDirectories_LocIndependent": [], "radDistDirectories_LocDependent": []}
    return e, tmp_path / "raddists" / "JET"


def test_loads_every_file_in_the_folder(run):
    e, base = run
    for i in range(3):
        write_raddist(base / "95709/elongatedRing/sub" / f"R_{i}.json")
    e.info["radDistDirectories_LocIndependent"] = ["95709/elongatedRing"]
    e._load_radDists()
    assert len(e.data["synthetic"]["locIndependent"]) == 3


def test_misspelled_folder_names_the_closest_match(run):
    e, base = run
    write_raddist(base / "95709/elongatedRing/R_0.json")
    e.info["radDistDirectories_LocIndependent"] = ["95709/elongatedRng"]
    with pytest.raises(FileNotFoundError) as err:
        e._load_radDists()
    msg = str(err.value)
    assert "95709/elongatedRng" in msg and "radDistDirectories_LocIndependent" in msg
    assert "Did you mean: 95709/elongatedRing" in msg


def test_missing_machine_folder_is_explained(run):
    e, base = run
    e.info["radDistDirectories_LocIndependent"] = ["95709/elongatedRing"]
    with pytest.raises(FileNotFoundError, match="does not exist"):
        e._load_radDists()


def test_empty_folder_is_an_error(run):
    e, base = run
    (base / "95709/elongatedRing").mkdir(parents=True)
    e.info["radDistDirectories_LocIndependent"] = ["95709/elongatedRing"]
    with pytest.raises(FileNotFoundError, match="no radDist files"):
        e._load_radDists()


def test_bad_files_are_skipped_with_a_warning(run):
    e, base = run
    write_raddist(base / "ring/R_0.json")
    (base / "ring/broken.json").write_text("{ not json")
    e.info["radDistDirectories_LocIndependent"] = ["ring"]
    with pytest.warns(UserWarning, match="broken.json"):
        e._load_radDists()
    assert len(e.data["synthetic"]["locIndependent"]) == 1


def test_folder_where_nothing_loads_is_an_error(run):
    e, base = run
    (base / "ring").mkdir(parents=True)
    (base / "ring/broken.json").write_text("{ not json")
    e.info["radDistDirectories_LocIndependent"] = ["ring"]
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="None of the 1 files"):
        e._load_radDists()


def test_only_one_folder_key_is_enough(run):
    e, base = run
    write_raddist(base / "helical/R_0.json")
    e.info = {"tokamakName": "JET", "radDistDirectories_LocDependent": ["helical"]}
    e._load_radDists()
    assert len(e.data["synthetic"]["locDependent"]["loc_0"]) == 1


def test_no_folders_at_all_is_an_error(run):
    e, _ = run
    with pytest.raises(RuntimeError, match="No radDist folders"):
        e._load_radDists()
