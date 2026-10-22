"""
tests/test_build_progress.py

What `emis3d make-raddists` prints: the plan before anything is built, one
line per radDist as it finishes (count, parameters, start position, time,
time left), and a summary; a failed radDist is reported and the rest still
build. The builders are replaced by quick stand-ins (no ray tracing).
"""

import re

import pytest
import yaml

import emis3d.util_radDist as Util_radDist
from emis3d.cli import main
from emis3d.pipeline import describe_shape, make_raddists

LIBRARY = {
    "tokamakName": "MOCK", "shot": 1, "folder": "lib", "injectionLocation": 225.0, "workers": 1,
    "rays": {"pixelSamples": 5},
    "grid": {"rLimits": [1.7, 1.9], "zLimits": [0.0, 0.0], "NumR": 2, "NumZ": 1},
    "types": {
        "ring": {"distType": "ElongatedRing", "sigma_R": [0.2, 0.3], "sigma_z": 0.25},
        "tube": {"distType": "SquareTube", "dR": 0.1, "dz": [0.1, 0.2]},
    },
}


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "emis3d.yaml").write_text("version: 1\n")
    (tmp_path / "raddists" / "MOCK").mkdir(parents=True)
    (tmp_path / "raddists" / "MOCK" / "raddists.yaml").write_text(yaml.safe_dump(LIBRARY))
    return tmp_path


@pytest.fixture
def fake_builders(monkeypatch):
    """Stand-ins that record what they were asked to build; one ring shape fails."""
    built = []

    def ring(job):
        rz, cfg = job
        if cfg["sigma_R"] == 0.3 and rz[0] > 1.8:
            raise RuntimeError("observation failed")
        built.append(("ring", cfg["sigma_R"], tuple(rz)))

    def tube(job):
        built.append(("tube", job[1]["dz"], tuple(job[0])))

    monkeypatch.setattr(Util_radDist, "radDist_ElongatedRing_parallel", ring)
    monkeypatch.setattr(Util_radDist, "radDist_SquareTube_parallel", tube)
    return built


def test_plan_comes_first_then_one_line_per_radDist_then_a_summary(ws, fake_builders, capsys):
    s = make_raddists(ws / "raddists" / "MOCK" / "raddists.yaml", workspace=ws, progress=True)
    out = capsys.readouterr().out.splitlines()

    first_progress = next(i for i, line in enumerate(out) if line.startswith("["))
    plan = "\n".join(out[:first_progress])
    assert "ring (ElongatedRing) -> lib/ring" in plan and "tube (SquareTube) -> lib/tube" in plan
    assert "sigma_R: 0.2, 0.3 | sigma_z: 0.25 | rotationAngle: 0" in plan
    assert "2 distinct shape(s) x 2 start position(s)" in plan and "= 4 radDists" in plan
    assert "Total: 8 radDists" in plan

    progress = [line for line in out if line.startswith("[")]
    assert len(progress) == 8
    assert re.match(r"\[1/8 +12\.5%\] ring: sigma_R 0\.2, sigma_z 0\.25, rotationAngle 0 \| R 1\.700, z 0\.000 \| \d+\.\d s \| elapsed", progress[0])
    assert progress[-1].startswith("[8/8 100.0%] tube: dR 0.1, dz 0.2")
    assert sum("FAILED: RuntimeError: observation failed" in line for line in progress) == 1

    assert "Built 7 of 8 radDists" in "\n".join(out) and "1 failed" in "\n".join(out)
    assert s["built"] == 7 and len(s["failed"]) == 1 and "sigma_R 0.3" in s["failed"][0]["parameters"]
    assert len(fake_builders) == 7                                                   # the rest still built


def test_failures_go_in_the_manifest_and_the_exit_code(ws, fake_builders, capsys):
    code = main(["make-raddists", "raddists.yaml", "--tokamak", "MOCK", "--workspace", str(ws)])
    assert code == 1
    manifest = yaml.safe_load((ws / "raddists" / "MOCK" / "lib" / "manifest.yaml").read_text())
    assert manifest["failed"][0]["error"] == "RuntimeError: observation failed"
    assert "Manifest:" in capsys.readouterr().out


def test_dry_run_prints_the_plan_and_builds_nothing(ws, fake_builders, capsys):
    assert main(["make-raddists", "raddists.yaml", "--tokamak", "MOCK", "--workspace", str(ws), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Total: 8 radDists" in out and "Dry run: nothing built" in out and "[" not in out.split("Total")[1]
    assert fake_builders == []


def test_quiet_by_default(ws, fake_builders, capsys):
    make_raddists(ws / "raddists" / "MOCK" / "raddists.yaml", workspace=ws, only=["tube"])
    assert capsys.readouterr().out == ""


def test_describe_shape_uses_the_config_names():
    assert describe_shape({"sigma_R": 0.05, "sigmaKernel": 0.01, "numFieldLines": 500, "numTransists": 1}) == (
        "sigma_R 0.05, sigmaKernel 0.01, numFieldLines 500, numTransits 1")
    assert describe_shape({"sourceFile": "/a/b/sim.json", "timestep": 40}) == "sim.json, timestep 40"


def test_workers_print_nothing_themselves(ws, capsys):
    """The builders used to print 'DONE with elongatedRing radDist' (even for other types)."""
    import inspect

    for name in ("ElongatedRing", "Helical", "HelicalRing", "SquareTube", "NIMROD", "M3DC1"):
        assert "print(" not in inspect.getsource(getattr(Util_radDist, f"radDist_{name}_parallel"))
