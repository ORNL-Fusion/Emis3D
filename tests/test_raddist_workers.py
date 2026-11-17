"""
tests/test_raddist_workers.py

The worker processes of make_raddists. They are started fresh ("spawn") on
every platform and handed the workspace itself, so radDists are built in
the same locations however the workspace was given. With forked workers
(Linux's default until Python 3.14) that came for free; with fresh ones
(macOS, Windows) only the workspace's folder reached a worker, through the
environment, and a location overridden for the run did not.
"""

import pickle

import pytest
import yaml

from emis3d import pipeline
from emis3d.workspace import Workspace, get_workspace, set_workspace

LIBRARY = {
    "tokamakName": "MOCK", "shot": 1, "folder": "lib", "injectionLocation": 225.0, "units": "Brightness",
    "workers": 2, "rays": {"pixelSamples": 5, "seed": 0},
    "grid": {"rLimits": [1.7, 1.9], "zLimits": [0.0, 0.0], "NumR": 2, "NumZ": 1},
    "types": {"ring": {"distType": "ElongatedRing", "sigma_R": 0.2, "sigma_z": 0.2}},
}


@pytest.fixture
def setup(tmp_path):
    """(workspace with radDists relocated by an override, library config, where the radDists must go)."""
    home, elsewhere = tmp_path / "ws", tmp_path / "scratch"
    (home / "raddists" / "MOCK").mkdir(parents=True)
    (home / "emis3d.yaml").write_text("version: 1\n")
    config = home / "raddists" / "MOCK" / "raddists.yaml"
    config.write_text(yaml.safe_dump(LIBRARY))
    ws = Workspace.resolve(workspace=home, overrides={"raddists": str(elsewhere / "{tokamak}")}, env={}, user_config=tmp_path / "none.yaml")
    return ws, config, elsewhere / "MOCK" / "lib" / "ring"


def test_a_worker_is_given_the_workspace(setup):
    ws, _, target = setup
    copy = pickle.loads(pickle.dumps(ws))                                       # as it reaches a fresh process
    assert copy.path("raddists", tokamak="MOCK") == ws.path("raddists", tokamak="MOCK") == target.parents[1]
    assert copy.run_id == ws.run_id
    try:
        set_workspace(None)
        pipeline._init_worker(copy)
        assert get_workspace() is copy
    finally:
        set_workspace(None)


def test_workers_are_started_fresh_with_the_workspace(setup, monkeypatch):
    ws, config, _ = setup
    seen = {}

    class InlineExecutor:
        """Stands in for ProcessPoolExecutor: records how it was set up and reports every radDist as built."""

        def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
            seen.update(workers=max_workers, method=mp_context.get_start_method(), initializer=initializer, initargs=initargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, task):
            from concurrent.futures import Future

            f = Future()
            f.set_result((0.0, None))
            return f

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", InlineExecutor)
    s = pipeline.make_raddists(config, workspace=ws)
    assert s["built"] == 2
    assert (seen["workers"], seen["method"]) == (2, "spawn")
    assert seen["initializer"] is pipeline._init_worker and seen["initargs"] == (ws,)


@pytest.mark.cherab
def test_parallel_build_writes_to_the_overridden_location(setup):
    ws, config, target = setup
    try:
        s = pipeline.make_raddists(config, workspace=ws)
    finally:
        set_workspace(None)
    assert s["built"] == 2 and s["failed"] == []
    assert len(list(target.rglob("*.json"))) == 2                               # where the override points
    assert not list((config.parent / "lib").rglob("R_*.json"))                  # not in the workspace's own folder
