"""
tests/test_checkpoints.py

Every time is saved on its own as soon as it is finished, so a run that
stops part of the way through (an error at a later time, a killed batch
job) keeps the times it finished. The results used to be written once, at
the very end.
"""

import dill
import numpy as np
import pytest

import emis3d.emis3D as emis3D_module
from emis3d.emis3D import Emis3D
from emis3d.pipeline import run
from test_calibration import write_workspace

TIMES = [9.5, 10.0, 10.5]


@pytest.fixture
def emis(tmp_path):
    e = Emis3D(runConfigName=str(write_workspace(tmp_path, tstart=9.5, tend=10.5, Nt=3)))
    return e


def finish(e, t):
    """What _perform_fit leaves for a time, in miniature (the real one needs ray tracing)."""
    if not hasattr(e, "bestFits"):
        e.bestFits, e.fitData = {}, {}
    e.bestFits[t] = {"bestFitID": 0, "radiation_distribution": {"toroidal_peaking_factor": 1.0 + t}}
    e.fitData[t] = {"observed": [[t]]}


def test_perform_fits_reports_each_finished_time(emis, monkeypatch):
    done = []
    monkeypatch.setattr(emis, "_perform_fit", lambda t, cleanup=False: done.append(("fit", t)))
    assert emis._perform_fits(after_each=lambda t: done.append(("after", t))) == TIMES
    assert done == [(kind, t) for t in TIMES for kind in ("fit", "after")]      # after each time, before the next
    done.clear()
    emis._perform_fits(evalTime=10.0, after_each=lambda t: done.append(("after", t)))
    assert done == [("fit", 10.0), ("after", 10.0)]
    emis._perform_fits(evalTime=10.0)                                             # optional


def test_a_checkpoint_holds_one_time_and_loads_like_a_results_file(emis):
    finish(emis, 9.5)
    path = emis._save_checkpoint(9.5)
    assert path == emis._results_dir() / "checkpoints" / "1_bestFits_9.5000.dill"
    with open(path, "rb") as f:
        saved = dill.load(f)
    assert list(saved["bestFits"]) == [9.5] and list(saved["fit_data"]) == [9.5]
    other = Emis3D(initialize=False)
    other._load_bestFits(path)
    assert other.bestFits[9.5]["radiation_distribution"]["toroidal_peaking_factor"] == 10.5


def test_an_interrupted_run_keeps_the_times_it_finished(emis, monkeypatch):
    def fit(t, cleanup=False):
        if t == 10.5:
            raise RuntimeError("Every fit failed at t=10.5")
        finish(emis, t)

    monkeypatch.setattr(emis, "_perform_fit", fit)
    with pytest.raises(RuntimeError, match="Every fit failed"):
        emis._perform_fits(after_each=emis._save_checkpoint)
    folder = emis._results_dir() / "checkpoints"
    assert sorted(p.name for p in folder.iterdir()) == ["1_bestFits_10.0000.dill", "1_bestFits_9.5000.dill"]   # no temporary files left

    recovered = Emis3D(initialize=False)
    recovered._load_bestFits(folder)                                              # the whole folder
    assert sorted(recovered.bestFits) == [9.5, 10.0] and sorted(recovered.fitData) == [9.5, 10.0]


def test_the_complete_file_replaces_the_checkpoints(emis):
    for t in TIMES:
        finish(emis, t)
        emis._save_checkpoint(t)
    path = emis._save_bestFits()
    assert path.name == "1_bestFits_9.5000_to_10.5000.dill" and path.is_file()
    assert not (emis._results_dir() / "checkpoints").exists()
    assert [p.name for p in emis._results_dir().glob(".*.tmp")] == []
    again = Emis3D(initialize=False)
    again._load_bestFits(path)
    assert sorted(again.bestFits) == TIMES


def test_loading_an_empty_folder_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .dill files"):
        Emis3D(initialize=False)._load_bestFits(tmp_path)


@pytest.mark.parametrize("save", [True, False])
def test_run_checkpoints_only_when_it_saves(monkeypatch, save):
    seen = {}

    class Fake:
        def __init__(self, runConfigName=None, workspace=None):
            self.info = {}

        def _save_checkpoint(self, t):
            pass

        def _perform_fits(self, evalTime=None, cleanup=None, after_each=None):
            seen.setdefault("after_each", []).append(after_each)

        def close(self):
            pass

        def _save_bestFits(self):
            seen["saved"] = True

    monkeypatch.setattr(emis3D_module, "Emis3D", Fake)
    run("run.yaml", plots=False, save=save)
    run("run.yaml", plots=False, save=save, times=[1.0, 2.0])
    assert len(seen["after_each"]) == 3                                           # once, then once per given time
    assert all((cb is not None) == save for cb in seen["after_each"])
    assert seen.get("saved", False) == save


@pytest.mark.cherab
def test_run_end_to_end_leaves_one_results_file(tmp_path):
    import yaml

    from emis3d.mock import selftest

    selftest(workdir=tmp_path, pixel_samples=20, verbose=False)
    cfg_path = tmp_path / "runs" / "selftest" / "run.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.update(tstart=9.8, tend=10.2, Nt=2)
    cfg_path.write_text(yaml.safe_dump(cfg))
    e = run(cfg_path, plots=False)
    out = e._results_dir()
    assert [p.name for p in out.glob("*.dill")] == ["1_bestFits_9.8000_to_10.2000.dill"] and not (out / "checkpoints").exists()
    assert np.isfinite(e.bestFits[9.8]["radiation_distribution"]["toroidal_peaking_factor"])
