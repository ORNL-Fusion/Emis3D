"""
tests/test_results_summary.py

The results of a run as plain numbers (emis3d.results): a JSON file written
beside the dill file, readable without Emis3D, lmfit or Cherab.
"""

import json
import types
from pathlib import Path

import numpy as np
import pytest

from emis3d import results as R
from emis3d.emis3D import Emis3D
from test_calibration import CAMS, write_workspace

T = 10.0
FIT = {"error_definition": "floor_relative", "scale_def": "exponential", "enable_dphi_scaling": True, "injectionLocation": 0.0,
       "uncertainty": {"n_samples": 200, "max_candidates": 3}}


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """A fitted and post-processed time on fabricated data; the radDist is not rebuilt (that needs the ray tracer's scene)."""
    root = tmp_path_factory.mktemp("ws")
    bolos = {c: {"dataFileName": f"1/{c}.h5"} for c in CAMS}
    bolos["POL_A"]["badChannels"] = ["POL_A02"]
    e = Emis3D(runConfigName=str(write_workspace(root, tstart=T, miscal={c: 1.0 for c in CAMS}, BOLOMETERS=bolos, **FIT)))
    e._prepare_fits(evalTime=T)
    e._minimize_radDists(evalTime=T)
    e._rebuild_radDist = lambda **kw: types.SimpleNamespace(info={"units": "Brightness"})
    e._post_process_fit_arrangement(evalTime=T)
    e._post_process_calculations(evalTime=T)
    e._post_process_uncertainty(evalTime=T)
    return e


def strict(text):
    """Parse as strict JSON: NaN and Infinity are not JSON."""
    def refuse(name):
        raise AssertionError(f"{name} is not JSON")
    return json.loads(text, parse_constant=refuse)


def test_to_plain():
    plain = R.to_plain({1: np.array([1.0, np.inf, np.nan]), "t": (np.float64(2.5), np.int64(3), np.bool_(True)), "p": Path("a/b"), "o": object()})
    assert plain == {"1": [1.0, None, None], "t": [2.5, 3, True], "p": "a/b", "o": None}
    assert json.dumps(plain, allow_nan=False)


def test_summary_has_the_fit_and_its_parameters(fitted):
    s = R.summary(fitted)
    assert (s["format"], s["tokamak"], s["shot"], s["times"]) == (R.FORMAT, "MOCK", 1, [T])
    assert s["settings"]["scale_def"] == "exponential" and s["run_config"].endswith("run.yaml")
    (r,) = s["results"]
    best = fitted.bestFits[T]
    fit = best["fit"]
    assert r["time"] == T and r["raddist"]["file"] == best["radDistInfo"]["radDistPath"] and r["raddist"]["distType"] == "elongatedRing"
    assert r["fit"]["chi2"] == pytest.approx(fit.chisqr) and r["fit"]["aic"] == pytest.approx(fit.aic) and r["fit"]["n_data"] == fit.ndata
    a = r["fit"]["parameters"]["a_0"]
    assert a["value"] == pytest.approx(fit.params["a_0"].value) and a["stderr"] == pytest.approx(fit.params["a_0"].stderr)
    assert (a["min"], a["max"], a["vary"]) == (fit.params["a_0"].min, fit.params["a_0"].max, True)
    assert set(r["fit"]["parameters"]) == set(fit.params)


def test_summary_has_every_cameras_data_and_model(fitted):
    (r,) = R.summary(fitted)["results"]
    assert list(r["cameras"]) == CAMS
    for g, cam in enumerate(CAMS):
        c = r["cameras"][cam]
        assert c["channels"] == fitted.channel_order["channel_list"][g]
        np.testing.assert_allclose(c["data"], fitted.fitData[T]["boloData"][cam])
        np.testing.assert_allclose(c["model"], fitted.bestFits[T]["synthData"]["elongatedRing"][cam])
        assert list(c["model_components"]) == ["elongatedRing"]
    a = r["cameras"]["POL_A"]
    k = a["channels"].index("POL_A02")
    assert a["fitted"][k] is False and a["error"][k] is None                       # the bad channel: infinite error, written as null
    assert all(a["fitted"][:k] + a["fitted"][k + 1:]) and all(e is not None for i, e in enumerate(a["error"]) if i != k)


def test_summary_has_the_profile_tpf_and_ranges(fitted):
    (r,) = R.summary(fitted)["results"]
    rad = fitted.bestFits[T]["radiation_distribution"]
    assert r["tpf"] == pytest.approx(rad["toroidal_peaking_factor"])
    np.testing.assert_allclose(r["toroidal_profile"]["power"], rad["total_power"])
    np.testing.assert_allclose(r["toroidal_profile"]["phi"], rad["phi"])
    assert r["quantities"]["total_power"] == pytest.approx(rad["quantities"]["total_power"])
    lo, hi = r["ranges"]["total"]["tpf"]["68%"]
    assert lo <= r["tpf"] <= hi
    assert len(r["ranges"]["profile_band"]["95%"][0]) == len(rad["phi"])


def test_save_writes_strict_json_beside_the_dill_file(fitted):
    path = fitted._save_bestFits()
    plain = path.with_suffix(".json")
    assert path.suffix == ".dill" and plain.is_file() and plain.stat().st_size < 0.2 * 1024**2
    data = strict(plain.read_text())
    assert data == R.load_summary(plain)
    assert data["results"][0]["fit"]["parameters"]["a_0"]["value"] == pytest.approx(fitted.bestFits[T]["fit"].params["a_0"].value)
    assert [p.name for p in path.parent.glob(".*.tmp")] == []


def test_a_time_without_post_processing_still_has_a_summary(fitted):
    """Parts a run does not have (here: everything but the fit) are left out, not an error."""
    bare = types.SimpleNamespace(info=fitted.info, bestFits={T: {"fit": fitted.bestFits[T]["fit"]}}, fitData={}, channel_order=None, run_config_path=None)
    (r,) = R.summary(bare)["results"]
    assert set(r) == {"time", "raddist", "fit", "at_bounds", "cameras"} and r["cameras"] == {}


def test_a_parametric_fit_is_summarized(fitted):
    from emis3d.parametric import ParametricResult

    ring = ParametricResult(model="ring", values={"startR": 1.8, "a": 1.1}, stderr={"startR": 0.01, "a": None}, chi2=12.0, n_data=40, aic=-30.0,
                            n_params=2, at_bounds=["sigma_R at its upper bound 1.5"], amplitude={"a": 950.0}, fit=object())
    shell = ParametricResult(model="shell", values={"psi_center": 0.8}, stderr={}, chi2=40.0, n_data=40, aic=5.0, n_params=3)
    best = {**fitted.bestFits[T], "parametric": ring, "parametric_models": [ring, shell]}
    emis = types.SimpleNamespace(info=fitted.info, bestFits={T: best}, fitData=fitted.fitData, channel_order=fitted.channel_order, run_config_path=None)
    (r,) = R.summary(emis)["results"]
    par = r["parametric"]
    assert par["model"] == "ring" and par["values"] == {"startR": 1.8, "a": 1.1} and par["stderr"]["a"] is None
    assert par["at_bounds"] == ["sigma_R at its upper bound 1.5"] and par["amplitude"] == {"a": 950.0}
    assert par["compared"] == [{"model": "ring", "aic": -30.0, "chi2": 12.0, "n_params": 2}, {"model": "shell", "aic": 5.0, "chi2": 40.0, "n_params": 3}]
    assert json.dumps(r, allow_nan=False)


@pytest.mark.cherab
def test_a_real_run_writes_the_summary(tmp_path):
    from emis3d.mock import selftest

    out = selftest(workdir=tmp_path, pixel_samples=20, verbose=False)
    (plain,) = list(Path(out["results"]).glob("*.json"))
    data = strict(plain.read_text())
    (r,) = data["results"]
    assert r["raddist"]["settings"]["startR"] == pytest.approx(1.75) and r["raddist"]["settings"]["sigma_R"] == pytest.approx(0.2)
    assert r["fit"]["parameters"]["b_elongatedRing_225"]["value"] == pytest.approx(out["b_fit"])
    assert set(r["cameras"]) == set(CAMS) and r["tpf"] > 1.0
