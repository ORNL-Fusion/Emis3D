"""
tests/test_parallel.py

Fitting in worker processes (util_emis3D.FitPool, used by
Emis3D._minimize_radDists with numProcessorsFitting > 1) gives the same
results as fitting serially. Uses the fabricated MOCK workspace.
"""

import numpy as np
import pytest

import emis3d.util_emis3D as U
from emis3d.emis3D import Emis3D
from test_calibration import CAMS, write_workspace

T = 10.0
FIT = {"error_definition": "floor_relative", "scale_def": "exponential", "enable_dphi_scaling": True}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    run = write_workspace(tmp_path_factory.mktemp("ws"), miscal={c: 1.0 for c in CAMS}, **FIT)
    e = Emis3D(runConfigName=str(run))
    e._prepare_fits(evalTime=T)
    return e, run


def jobs_for(e, repeat=1):
    base = [(ii, e.fits[T][ii]["parameters"], e.fitData[T], e.fits[T][ii]["synthetic_dict"], "exponential", 0.0, "least_squares")
            for ii in e.fits[T] if isinstance(ii, int)]
    return [(k,) + j[1:] for k, j in enumerate(base * repeat)]


def test_worker_processes_give_the_serial_results(prepared, monkeypatch):
    e, _ = prepared
    jobs = jobs_for(e, repeat=4)
    serial = U.FitPool(1).map(jobs)
    monkeypatch.setattr(U.FitPool, "MIN_JOBS_PER_WORKER", 1)
    pool = U.FitPool(2)
    try:
        parallel = pool.map(jobs)
        assert pool._executor is not None                         # really ran in processes
    finally:
        pool.close()
    assert [r[0] for r in parallel] == [j[0] for j in jobs]       # job order kept
    for (_, a, _), (_, b, err) in zip(serial, parallel, strict=True):
        assert err is None
        assert U.fit_score(a) == pytest.approx(U.fit_score(b), abs=1e-12)
        assert b.covar is not None                                # covariance survives the trip back


def test_a_numerical_failure_is_reported_not_raised(prepared):
    """A fit that fails numerically (here a NaN in the model) is recorded, and the others carry on."""
    import copy

    e, _ = prepared
    bad = list(jobs_for(e)[0])
    synth = copy.deepcopy(bad[3])
    name = synth["emissionNames"][0]
    synth[name]["data"][0][0] = [np.nan]
    bad[3] = synth
    index, fit, error = U.FitPool(1).map([tuple(bad)])[0]
    assert index == 0 and fit is None and "ValueError" in error


def test_a_programming_error_propagates(prepared):
    """A malformed synthetic dict is a bug, not a failed fit: it must not be hidden."""
    e, _ = prepared
    bad = list(jobs_for(e)[0])
    bad[3] = {"emissionNames": ["missing"]}
    with pytest.raises(KeyError):
        U.FitPool(1).map([tuple(bad)])


def test_small_batches_do_not_start_processes(prepared):
    e, _ = prepared
    pool = U.FitPool(4)
    pool.map(jobs_for(e))                                         # 3 fits, far below 50 per worker
    assert pool._executor is None


def test_emis3d_uses_the_pool_and_reuses_it(prepared, monkeypatch):
    e, run = prepared
    e.info["numProcessorsFitting"] = 1
    e._minimize_radDists(evalTime=T)
    serial = np.array(e.fits[T]["scores"], dtype=float)

    monkeypatch.setattr(U.FitPool, "MIN_JOBS_PER_WORKER", 1)
    e.info["numProcessorsFitting"] = 2
    try:
        e._minimize_radDists(evalTime=T)
        executor = e._fit_pool._executor
        assert executor is not None
        np.testing.assert_allclose(np.array(e.fits[T]["scores"], dtype=float), serial, rtol=0, atol=1e-12)
        e._minimize_radDists(evalTime=T)
        assert e._fit_pool._executor is executor                  # same workers, no second start-up
    finally:
        e.close()
    assert e._fit_pool is None
