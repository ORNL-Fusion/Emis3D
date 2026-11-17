"""
tests/test_time_grid_warning.py

tend after tstart without Nt. Nt defaults to 1, so only tstart is fitted
and the rest of the range silently is not: a config that reads "from 2100
to 2130" fits one time. The behaviour is unchanged (it is documented), but
it is now said: a warning when the config is loaded, and a finding in
`emis3d check`.
"""

import pytest

from emis3d.emis3D import Emis3D
from emis3d.pipeline import check
from emis3d.run_config import RunConfig
from test_calibration import write_workspace

BASE = {"BOLOMETERS": {"A": {"dataFileName": "a.h5"}}, "shot": 1, "radDistDirectories_LocIndependent": ["x"]}


@pytest.mark.parametrize("times, nt_text", [({"tstart": 1.0, "tend": 2.0}, "Nt is not set"), ({"tstart": 1.0, "tend": 2.0, "Nt": 1}, "Nt is 1")])
def test_a_range_with_one_time_is_a_problem(times, nt_text):
    cfg = RunConfig.from_dict({**BASE, **times})
    assert cfg.eval_times() == [1.0]                                           # unchanged: tstart alone
    msg = cfg.time_grid_problem()
    assert "tend (2.0) is after tstart (1.0)" in msg and nt_text in msg and "only t = 1.0 is fitted" in msg and "Set Nt" in msg


@pytest.mark.parametrize("times", [{}, {"tstart": 1.0}, {"tstart": 1.0, "tend": 1.0}, {"tstart": 1.0, "tend": 1.0, "Nt": 10},
                                   {"tstart": 1.0, "tend": 2.0, "Nt": 2}, {"tstart": 1.0, "tend": 2.0, "Nt": 11}])
def test_times_that_are_fitted_as_they_read_are_not(times):
    assert RunConfig.from_dict({**BASE, **times}).time_grid_problem() is None


def test_loading_the_config_warns(tmp_path, caplog):
    run = write_workspace(tmp_path, tstart=9.5, tend=10.5)
    with caplog.at_level("WARNING", logger="emis3d.emis3D"):
        e = Emis3D(runConfigName=str(run))
    assert e.eval_times() == [9.5]
    assert any("Evaluation times" in r.getMessage() and "only t = 9.5 is fitted" in r.getMessage() for r in caplog.records)


def test_a_complete_range_does_not_warn(tmp_path, caplog):
    run = write_workspace(tmp_path, tstart=9.5, tend=10.5, Nt=3)
    with caplog.at_level("WARNING", logger="emis3d.emis3D"):
        Emis3D(runConfigName=str(run))
    assert not any("Evaluation times" in r.getMessage() for r in caplog.records)


def test_check_reports_it(tmp_path):
    found = check(write_workspace(tmp_path, tstart=9.5, tend=10.5))
    warnings = [f.message for f in found if f.level == "warning"]
    assert any("evaluation times" in m and "only t = 9.5 is fitted" in m for m in warnings)
    assert not any(f.level == "error" for f in found)
