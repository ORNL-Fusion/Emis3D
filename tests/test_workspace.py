"""
tests/test_workspace.py

Where Emis3D finds machines and data (src/emis3d/workspace.py) and the `emis3d`
command (src/emis3d/cli.py). Every test passes its own environment and a
non-existent user config, so nothing on the developer's machine leaks in.
"""

from pathlib import Path

import pytest
import yaml

from emis3d import cli
from emis3d.workspace import (
    BUILTIN_MACHINES,
        Workspace,
    WorkspaceError,
)


@pytest.fixture
def no_user(tmp_path):
    return tmp_path / "no_user_config.yaml"


def make_ws(tmp_path, workspace_yaml=None):
    ws = tmp_path / "ws"
    (ws / "runs" / "184407").mkdir(parents=True)
    if workspace_yaml is not None:
        (ws / "emis3d.yaml").write_text(yaml.safe_dump(workspace_yaml))
    return ws


# ---------------------------------------------------------------------------
# Choosing the layout
# ---------------------------------------------------------------------------

def test_nothing_configured_explains_how_to_fix(no_user):
    ws = Workspace.resolve(env={}, user_config=no_user)
    assert ws.layout == "unconfigured"
    with pytest.raises(WorkspaceError, match="emis3d init"):
        ws.path("raddists", tokamak="MOCK")
    # the built-in MOCK machine works without any configuration
    assert ws.has_machine("MOCK")
    assert ws.machines() == ["MOCK"]


def test_old_root_variable_is_ignored(tmp_path, no_user):
    ws = Workspace.resolve(env={"EMIS3D_ROOT": str(tmp_path)}, user_config=no_user)
    assert ws.layout == "unconfigured"


def test_workspace_from_env_uses_new_layout(tmp_path, no_user):
    wsdir = make_ws(tmp_path)
    ws = Workspace.resolve(env={"EMIS3D_RUNS_DIR": str(wsdir)}, user_config=no_user)
    assert ws.layout == "workspace"
    assert ws.path("measurements", tokamak="DIII-D") == wsdir / "measurements/DIII-D"
    assert ws.path("raddists", tokamak="JET") == wsdir / "raddists/JET"
    assert ws.machine_roots() == [wsdir / "machines", BUILTIN_MACHINES]


def test_results_template_needs_its_fields(tmp_path, no_user):
    wsdir = make_ws(tmp_path)
    ws = Workspace.resolve(workspace=wsdir, user_config=no_user)
    with pytest.raises(WorkspaceError, match="run_dir"):
        ws.path("results", tokamak="DIII-D")
    run_dir = wsdir / "runs" / "184407"
    d = ws.results_dir(tokamak="DIII-D", run_dir=str(run_dir))
    assert d == run_dir / "results" / ws.run_id and d.is_dir()


def test_user_default_workspace(tmp_path):
    wsdir = make_ws(tmp_path)
    user = tmp_path / "user.yaml"
    user.write_text(yaml.safe_dump({"default_workspace": str(wsdir)}))
    ws = Workspace.resolve(env={}, user_config=user)
    assert ws.fields["workspace"] == str(wsdir.resolve())


# ---------------------------------------------------------------------------
# Precedence and templates
# ---------------------------------------------------------------------------

def test_precedence_user_workspace_run_explicit(tmp_path):
    wsdir = make_ws(tmp_path, {"paths": {"raddists": "from_workspace/{tokamak}",
                                         "cache": "from_workspace_cache"}})
    user = tmp_path / "user.yaml"
    user.write_text(yaml.safe_dump({"paths": {"raddists": "from_user", "cache": "from_user",
                                              "equilibria": "from_user/{tokamak}"}}))
    ws = Workspace.resolve(workspace=wsdir, user_config=user,
                           run_paths={"cache": "from_run"},
                           overrides={"equilibria": "/explicit/{tokamak}"})
    assert ws.path("raddists", tokamak="T") == wsdir / "from_workspace/T"       # workspace beats user
    assert ws.path("cache") == wsdir / "from_run"                               # run beats workspace
    assert ws.path("equilibria", tokamak="T") == Path("/explicit/T")           # explicit beats all
    assert ws.sources["equilibria"] == "argument"


def test_run_config_finds_workspace_above_it(tmp_path, no_user):
    wsdir = make_ws(tmp_path, {"version": 1})
    other = make_ws(tmp_path / "other")
    run = wsdir / "runs" / "184407" / "run.yaml"
    run.write_text("shot: 184407\n")
    ws = Workspace.resolve(run_config=run, env={"EMIS3D_RUNS_DIR": str(other)}, user_config=no_user)
    assert ws.fields["workspace"] == str(wsdir.resolve())


def test_per_tokamak_templates(tmp_path, no_user):
    wsdir = make_ws(tmp_path, {"tokamaks": {"JET": {"measurements": "/work/jet/sxr"}}})
    ws = Workspace.resolve(workspace=wsdir, user_config=no_user)
    assert ws.path("measurements", tokamak="JET") == Path("/work/jet/sxr")
    assert ws.path("measurements", tokamak="DIII-D") == wsdir / "measurements/DIII-D"


def test_env_vars_and_home_in_templates(tmp_path, no_user, monkeypatch):
    monkeypatch.setenv("E3D_SCRATCH", str(tmp_path / "scratch"))
    wsdir = make_ws(tmp_path, {"paths": {"raddists": "${E3D_SCRATCH}/{tokamak}", "cache": "~/e3dcache"}})
    ws = Workspace.resolve(workspace=wsdir, user_config=no_user)
    assert ws.path("raddists", tokamak="DIII-D") == tmp_path / "scratch/DIII-D"
    assert ws.path("cache") == Path.home() / "e3dcache"


def test_unset_env_var_is_named(tmp_path, no_user, monkeypatch):
    monkeypatch.delenv("E3D_NOT_SET", raising=False)
    wsdir = make_ws(tmp_path, {"paths": {"cache": "${E3D_NOT_SET}/cache"}})
    ws = Workspace.resolve(workspace=wsdir, user_config=no_user)
    with pytest.raises(WorkspaceError, match="E3D_NOT_SET"):
        ws.path("cache")


@pytest.mark.parametrize("bad", [{"pathz": {}}, {"paths": {"raddist": "x"}}])
def test_typos_are_errors(tmp_path, no_user, bad):
    wsdir = make_ws(tmp_path, bad)
    with pytest.raises(WorkspaceError, match="Unknown"):
        Workspace.resolve(workspace=wsdir, user_config=no_user)


# ---------------------------------------------------------------------------
# Machine search path
# ---------------------------------------------------------------------------

def test_user_camera_file_overrides_builtin(tmp_path, no_user):
    wsdir = make_ws(tmp_path)
    cam = wsdir / "machines/MOCK/inputs/sxrInfo/POL_A_config.yaml"
    cam.parent.mkdir(parents=True)
    cam.write_text("NAME: POL_A\n")
    ws = Workspace.resolve(workspace=wsdir, user_config=no_user)
    assert ws.machine_file("MOCK", "inputs", "sxrInfo", "POL_A_config.yaml") == cam
    # everything not overridden still comes from the built-in machine
    assert ws.machine_file("MOCK", "MOCK_settings.yaml").is_relative_to(BUILTIN_MACHINES)


def test_workspace_machine_is_listed(tmp_path, no_user):
    wsdir = make_ws(tmp_path)
    (wsdir / "machines/MYTOK").mkdir(parents=True)
    (wsdir / "machines/MYTOK/MYTOK_settings.yaml").write_text("MACHINE: {}\n")
    ws = Workspace.resolve(workspace=wsdir, user_config=no_user)
    assert ws.machines() == ["MOCK", "MYTOK"]
    with pytest.raises(FileNotFoundError, match="Looked in"):
        ws.machine_file("MYTOK", "inputs", "missing.txt")


def test_tokamak_uses_user_camera_override(tmp_path, no_user):
    """A real Tokamak build picks up a camera file from the workspace."""
    from emis3d.Tokamak import Tokamak

    wsdir = make_ws(tmp_path)
    builtin = BUILTIN_MACHINES / "MOCK/inputs/sxrInfo/POL_A_config.yaml"
    cam = wsdir / "machines/MOCK/inputs/sxrInfo/POL_A_config.yaml"
    cam.parent.mkdir(parents=True)
    cam.write_text(builtin.read_text().replace("[2.3, 0.8, 45.0]", "[2.3, 0.8, 50.0]"))
    ws = Workspace.resolve(workspace=wsdir, user_config=no_user)
    tok = Tokamak(tokamakName="MOCK", mode="Analysis", loadBolometers=True, workspace=ws)
    phis = {b.name: b.info["CAMERA_POSITION_R_Z_PHI"][2] for b in tok.bolometers}
    assert phis["POL_A"] == 50.0
    assert phis["POL_C"] == 285.0                       # untouched cameras come from the built-in machine
    assert tok.wall is not None


def test_camera_config_can_live_anywhere(tmp_path, no_user, monkeypatch):
    """configFileName may be an absolute or ~ path instead of a name in sxrInfo/."""
    from emis3d.Tokamak import Tokamak

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    cams = tmp_path / "home" / "cams"
    cams.mkdir(parents=True)
    builtin = BUILTIN_MACHINES / "MOCK"
    (cams / "POL_A.yaml").write_text(
        (builtin / "inputs/sxrInfo/POL_A_config.yaml").read_text().replace("[2.3, 0.8, 45.0]", "[2.3, 0.8, 55.0]"))
    (cams / "POL_C.yaml").write_text(
        (builtin / "inputs/sxrInfo/POL_C_config.yaml").read_text().replace("[2.3, 0.8, 285.0]", "[2.3, 0.8, 290.0]"))
    wsdir = make_ws(tmp_path)
    settings = wsdir / "machines/MOCK/MOCK_settings.yaml"
    settings.parent.mkdir(parents=True)
    settings.write_text((builtin / "MOCK_settings.yaml").read_text()
                        .replace("configFileName: POL_A_config.yaml", "configFileName: ~/cams/POL_A.yaml")
                        .replace("configFileName: POL_C_config.yaml", f"configFileName: {cams / 'POL_C.yaml'}"))
    ws = Workspace.resolve(workspace=wsdir, user_config=no_user)
    tok = Tokamak(tokamakName="MOCK", mode="Analysis", loadBolometers=True, workspace=ws)
    phis = {b.name: b.info["CAMERA_POSITION_R_Z_PHI"][2] for b in tok.bolometers}
    assert phis["POL_A"] == 55.0 and phis["POL_C"] == 290.0 and phis["POL_B"] == 165.0


def test_unknown_machine_lists_what_exists(no_user):
    from emis3d.Tokamak import Tokamak

    ws = Workspace.resolve(env={}, user_config=no_user)
    with pytest.raises(ValueError, match="MOCK"):
        Tokamak(tokamakName="DIII-D", workspace=ws)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def test_cli_init_and_paths(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("emis3d.workspace.USER_CONFIG", tmp_path / "none.yaml")
    target = tmp_path / "emis3d-runs"
    assert cli.main(["init", str(target)]) == 0
    for d in ("machines", "equilibria", "measurements", "raddists", "cache"):
        assert (target / d).is_dir()
    data = yaml.safe_load((target / "emis3d.yaml").read_text())
    assert data["paths"]["raddists"] == "raddists/{tokamak}"
    assert (target / "runs/example/run.yaml").is_file()

    # the example run finds its workspace by itself
    capsys.readouterr()
    assert cli.main(["paths", "--config", str(target / "runs/example/run.yaml")]) == 0
    out = capsys.readouterr().out
    assert "layout: workspace" in out
    assert str(target / "measurements" / "MOCK") in out
    assert "built-in" in out

    assert cli.main(["init", str(target)]) == 1          # refuses to overwrite


def test_cli_init_set_default(tmp_path, monkeypatch):
    user = tmp_path / "cfg" / "config.yaml"
    monkeypatch.setattr(cli, "USER_CONFIG", user)
    assert cli.main(["init", str(tmp_path / "w"), "--set-default"]) == 0
    assert yaml.safe_load(user.read_text())["default_workspace"] == str((tmp_path / "w").resolve())


def test_cli_machines(capsys, monkeypatch):
    monkeypatch.delenv("EMIS3D_RUNS_DIR", raising=False)
    monkeypatch.setattr("emis3d.workspace.USER_CONFIG", Path("/nonexistent/config.yaml"))
    assert cli.main(["machines"]) == 0
    assert "MOCK" in capsys.readouterr().out
