"""
tests/test_eval_times.py

Evaluation times come from the run config (tstart, tend, Nt), and
_perform_fits() / _plot_bestFit() loop over them.
"""

import pytest
import yaml

from emis3d.emis3D import Emis3D
from emis3d.run_config import RunConfig
from test_calibration import CAMS, write_workspace

BASE = {"BOLOMETERS": {}, "shot": 1, "radDistDirectories_LocIndependent": ["x"]}


@pytest.mark.parametrize("times, expected", [
    ({}, []),                                                       # no times configured
    ({"tstart": 50.949}, [50.949]),
    ({"tstart": 50.949, "tend": 50.949, "Nt": 10}, [50.949]),        # one time, whatever Nt is
    ({"tstart": 50.949, "tend": 50.953, "Nt": 5}, [50.949, 50.950, 50.951, 50.952, 50.953]),
    ({"tstart": 1.0, "tend": 2.0}, [1.0]),                           # Nt defaults to 1
])
def test_eval_times(times, expected):
    got = RunConfig.from_dict({**BASE, **times}).eval_times()
    assert got == expected                                          # exact, so usable as dict keys


@pytest.mark.parametrize("times, match", [
    ({"tstart": 2.0, "tend": 1.0}, "before tstart"),
    ({"tstart": 1.0, "tend": 2.0, "Nt": 0}, "Nt"),
    ({"tstart": 1.0, "tend": 2.0, "Nt": 2.5}, "Nt"),
    ({"tend": 2.0}, "need tstart"),
])
def test_bad_times_are_rejected(times, match):
    with pytest.raises(ValueError, match=match):
        RunConfig.from_dict({**BASE, **times})


@pytest.fixture
def run3(tmp_path):
    return write_workspace(tmp_path, miscal={c: 1.0 for c in CAMS}, tstart=9.5, tend=10.5, Nt=3)


def test_perform_fits_loops_over_the_configured_times(run3, monkeypatch):
    e = Emis3D(runConfigName=str(run3))
    calls = []
    monkeypatch.setattr(e, "_perform_fit", lambda t, cleanup=False: calls.append((t, cleanup)))
    assert e._perform_fits() == [9.5, 10.0, 10.5]
    assert calls == [(9.5, True), (10.0, True), (10.5, True)]       # loops free each time's other fits


def test_a_single_time_still_works_and_keeps_its_fits(run3, monkeypatch):
    e = Emis3D(runConfigName=str(run3))
    calls = []
    monkeypatch.setattr(e, "_perform_fit", lambda t, cleanup=False: calls.append((t, cleanup)))
    assert e._perform_fits(evalTime=10.2) == [10.2]
    assert calls == [(10.2, False)]


def test_no_configured_times_is_a_clear_error(tmp_path):
    e = Emis3D(runConfigName=str(write_workspace(tmp_path)))
    with pytest.raises(ValueError, match="tstart"):
        e._perform_fits()


def test_time_outside_the_data_is_an_error(run3):
    e = Emis3D(runConfigName=str(run3))                             # data run from 9 to 11
    with pytest.raises(ValueError, match="outside .* data"):
        e._prepare_fits(evalTime=12.0)


def test_plot_best_fit_loops_over_the_fitted_times(run3, monkeypatch):
    e = Emis3D(runConfigName=str(run3))
    e.bestFits = {10.5: {}, 9.5: {}, 10.0: {}}
    calls = []
    monkeypatch.setattr(e, "_plot_bestFit_one", lambda t, save=False: calls.append((t, save)))
    e._plot_bestFit(save=True)
    assert calls == [(9.5, True), (10.0, True), (10.5, True)]


@pytest.mark.cherab
def test_three_times_end_to_end(tmp_path):
    from emis3d.mock import selftest

    selftest(workdir=tmp_path, pixel_samples=20, verbose=False)
    run = tmp_path / "runs" / "selftest" / "run.yaml"
    cfg = yaml.safe_load(run.read_text())
    cfg.update(tstart=9.6, tend=10.4, Nt=3)
    run.write_text(yaml.safe_dump(cfg))

    e = Emis3D(runConfigName=str(run))
    try:
        assert e._perform_fits() == [9.6, 10.0, 10.4]
    finally:
        e.close()
    assert sorted(e.bestFits) == [9.6, 10.0, 10.4]
    for t in e.bestFits:
        assert "ranges" in e.bestFits[t]
        assert [k for k in e.fits[t] if isinstance(k, int)] == [e.bestFits[t]["bestFitID"]]   # others freed
    e._plot_bestFit(save=True)
    pngs = sorted(p.name for p in (tmp_path / "runs" / "selftest" / "results").glob("*/images/*.png"))
    assert pngs == ["1_10.0000.png", "1_10.4000.png", "1_9.6000.png"]
