"""
tests/test_parallel_times.py

A parametric run fits its times in worker processes when numProcessorsFitting
(or run's workers) is above 1 (pipeline._fit_times_in_parallel): the results
are those of the serial loop, a failing time keeps the finished ones, and the
field-line mappings are built before the workers start.
"""

import logging
from types import SimpleNamespace

import dill
import numpy as np
import pytest

import emis3d.pipeline as pipeline
from emis3d.geometry import matrix_path
from emis3d.pipeline import run
from emis3d.tokamak import Tokamak
from emis3d.workspace import Workspace, set_workspace
from test_calibration import write_workspace
from test_parametric import matrix_for

PARAMETRIC = {"models": ["ring"], "matrices": ["low"], "maxiter": 5, "bounds": {"ring": {"startR": [1.6, 1.9], "startZ": [0.0, 0.3]}}}


def parametric_workspace(root, **config):
    """A MOCK workspace with data from 9 to 11, a synthetic matrix named low, and a parametric run config."""
    root.mkdir(exist_ok=True)
    settings = {"tstart": 9.5, "tend": 10.5, "Nt": 3, "radDistDirectories_LocIndependent": [], "solver": "parametric",
                "scale_def": "exponential", "error_definition": "floor_relative", "parametric": PARAMETRIC}
    cfg = write_workspace(root, **{**settings, **config})
    ws = Workspace.resolve(workspace=root, env={}, user_config=root / "none.yaml")
    set_workspace(ws)
    try:
        tokamak = Tokamak(tokamakName="MOCK", mode="Analysis", workspace=ws)
    finally:
        set_workspace(None)
    matrix_for((None, SimpleNamespace(tokamak=tokamak), None)).save(matrix_path(ws, "MOCK", "low"))
    return cfg


def numbers(e):
    """What a run found at every time, as plain numbers."""
    out = {}
    for t, best in e.bestFits.items():
        rad = best["radiation_distribution"]
        out[t] = (best["parametric"].model, dict(best["parametric"].values), rad["toroidal_peaking_factor"], np.asarray(rad["total_power"]),
                  {name: par.value for name, par in best["fit"].params.items()}, best["ranges"]["total"].get("tpf"))
    return out


def test_parallel_times_give_the_serial_results(tmp_path, capsys):
    serial = numbers(run(parametric_workspace(tmp_path / "serial"), workspace=tmp_path / "serial", workers=1, plots=False, save=False))
    capsys.readouterr()

    e = run(parametric_workspace(tmp_path / "parallel"), workspace=tmp_path / "parallel", workers=2, plots=False)
    out = capsys.readouterr().out
    assert "Fitting 3 times in 2 worker processes" in out
    assert out.count("fitted in") == 3 and "[3/3]" in out                    # one line per finished time
    assert "Best Fit" not in out                                             # the workers' own reports are dropped

    parallel = numbers(e)
    assert list(parallel) == [9.5, 10.0, 10.5] == list(serial)                # in the order of the times
    for t in serial:
        model, values, tpf, power, params, tpf_range = serial[t]
        assert parallel[t][0] == model
        assert parallel[t][1] == values and parallel[t][4] == params        # the same fit, to the last digit
        assert parallel[t][2] == tpf and np.array_equal(parallel[t][3], power)
        assert parallel[t][5] == tpf_range
    assert sorted(e.fitData) == [9.5, 10.0, 10.5]

    # --- Saved as a serial run saves: one file with every time, and no checkpoints left
    saved = list((tmp_path / "parallel").rglob("1_bestFits_*.dill"))
    assert [p.name for p in saved] == ["1_bestFits_9.5000_to_10.5000.dill"]
    assert not list((tmp_path / "parallel").rglob("checkpoints"))
    with open(saved[0], "rb") as f:
        assert sorted(dill.load(f)["bestFits"]) == [9.5, 10.0, 10.5]


def test_without_save_no_file_is_left(tmp_path):
    e = run(parametric_workspace(tmp_path), workspace=tmp_path, workers=2, times=[10.0, 10.2], plots=False, save=False)
    assert sorted(e.bestFits) == [10.0, 10.2]
    assert not list(tmp_path.rglob("*.dill"))


def test_a_failing_time_raises_and_the_finished_time_is_kept(tmp_path, capsys):
    cfg = parametric_workspace(tmp_path)
    with pytest.raises(Exception, match="99"):                                # there is no data at t = 99
        run(cfg, workspace=tmp_path, workers=2, times=[10.0, 99.0], plots=False)
    assert "A time failed" in capsys.readouterr().err
    kept = list(tmp_path.rglob("checkpoints/*.dill"))
    assert [p.name for p in kept] == ["1_bestFits_10.0000.dill"]
    with open(kept[0], "rb") as f:
        assert list(dill.load(f)["bestFits"]) == [10.0]


@pytest.mark.parametrize("config, workers, times, parallel", [
    ({}, 2, None, True),                                                     # parametric, three times, two workers
    ({"numProcessorsFitting": 3}, None, None, True),                         # the run config's own setting
    ({"numProcessorsFitting": 3}, 1, None, False),                           # ... which workers overrides
    ({}, 1, None, False),                                                    # one worker
    ({}, 2, [10.0], False),                                                  # one time
    ({"solver": "library", "radDistDirectories_LocIndependent": ["ring"], "scale_def": "constant"}, 2, None, False),  # library: FitPool's job
])
def test_only_a_parametric_run_with_several_times_and_workers_is_split(tmp_path, monkeypatch, config, workers, times, parallel):
    from emis3d.emis3D import Emis3D

    calls = []
    monkeypatch.setattr(pipeline, "_fit_times_in_parallel", lambda e, cfg, todo, n, keep_checkpoints: calls.append(("parallel", todo, n, keep_checkpoints)))
    monkeypatch.setattr(Emis3D, "_perform_fits", lambda self, evalTime=None, cleanup=None, after_each=None: calls.append(("serial", evalTime)))
    run(parametric_workspace(tmp_path, **config), workspace=tmp_path, workers=workers, times=times, plots=False, save=False)
    if parallel:
        assert calls == [("parallel", [9.5, 10.0, 10.5], workers or 3, False)]
    else:
        assert calls and all(kind == "serial" for kind, *_ in calls)


def test_mappings_are_built_once_before_the_workers_start(tmp_path, monkeypatch):
    from emis3d.emis3D import Emis3D
    import emis3d.fieldline_map as fieldline_map

    # --- Only field-aligned models need a mapping: one per matrix, however many models use it
    built = []
    monkeypatch.setattr(fieldline_map, "get_or_build", lambda ws, G, machine, name, *a, **k: (built.append(name), (None, tmp_path / f"{name}_map.npz"))[1])
    cfg = parametric_workspace(tmp_path, parametric={**PARAMETRIC, "equilibrium": "MOCK_solovev.geqdsk"})
    e = Emis3D(runConfigName=str(cfg), workspace=tmp_path)
    e._build_parametric_mappings()
    assert built == []                                                       # a ring has none
    e.config.parametric["models"] = ["ring", "helical", "pair"]
    e.config.parametric["composites"] = {"pair": {"components": [{"model": "helical_psi", "name": "band"}, {"model": "ring", "name": "ring"}]}}
    e._build_parametric_mappings()
    assert built == ["low"]

    # --- ... and that happens before any worker process exists
    order = []

    class NoPool:
        def __init__(self, *args, **kwargs):
            order.append("workers")
            raise RuntimeError("stop here")

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", NoPool)
    fake = SimpleNamespace(_build_parametric_mappings=lambda: order.append("mappings"), ws=None)
    with pytest.raises(RuntimeError, match="stop here"):
        pipeline._fit_times_in_parallel(fake, "run.yaml", [1.0, 2.0], 2, keep_checkpoints=True)
    assert order == ["mappings", "workers"]


def test_a_workers_warning_names_its_time(monkeypatch):
    record = logging.LogRecord("emis3d.parametric", logging.WARNING, __file__, 1, "ring fit: %s at its upper bound", ("startR",), None)
    monkeypatch.setitem(pipeline._TIME_WORKER, "time", 10.5)
    assert pipeline._TimePrefix().filter(record)
    assert record.getMessage() == "[t = 10.5] ring fit: startR at its upper bound"
