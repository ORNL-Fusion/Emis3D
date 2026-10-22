"""
tests/test_fit_errors.py

Review issue 11: what the fit minimizes and how radDists are ranked.

- channel_error: floor_relative errors (additive floor + relative term), so
  strong channels dominate without near-zero channels taking over
- bad channels: listed in the run config, zero for the whole shot, or missing;
  left out of the fit entirely
- residual: signed, bad channels excluded, near-zero and negative data kept
- fit_radDist / fit_score: one fit and one ranking (AIC) for the class and
  the script

Uses the fabricated MOCK workspace from test_calibration (no ray tracing).
"""

import h5py
import numpy as np
import pytest
from lmfit import Parameters

import emis3d.util_emis3D as U
from emis3d.emis3D import Emis3D
from emis3d.run_config import RunConfig
from test_calibration import CAMS, TRUE_RD, cam_info, write_workspace

FIT = {"error_definition": "floor_relative", "scale_def": "constant", "error_floor": 0.05, "error_relative": 0.05}


# ---------------------------------------------------------------------------
# Channel errors
# ---------------------------------------------------------------------------

def test_floor_relative_formula():
    sig = U.channel_error("floor_relative", [0.0, 1.0, 10.0], peak=10.0, floor=0.05, relative=0.1)
    assert sig == pytest.approx(np.sqrt(0.5**2 + (0.1 * np.array([0.0, 1.0, 10.0])) ** 2))


def test_measured_noise_raises_the_floor_per_channel():
    sig = U.channel_error("floor_relative", [0.0, 0.0], peak=10.0, floor=0.05, relative=0.0, noise=[0.1, 2.0])
    assert sig == pytest.approx([0.5, 2.0])                  # floor 0.5 kept where the noise is lower


def test_older_models_keep_their_values_and_skip_non_positive_data():
    d = np.array([-1.0, 0.0, 2.0, 8.0])
    sig = U.channel_error("exponential", d, peak=8.0)
    assert np.all(np.isinf(sig[:2]))
    assert sig[2:] == pytest.approx(d[2:] * U.signal_error("exponential", d[2:], 8.0))


def test_strong_channels_dominate_without_near_zero_taking_over():
    """A 2%-of-peak leak into a near-zero channel costs less than a 5% miss at the peak."""
    peak, sig = 1.0, lambda d: U.channel_error("floor_relative", [d], peak)[0]
    near_zero = ((0.03 - 0.01) / sig(0.01)) ** 2
    at_peak = (0.05 / sig(1.0)) ** 2
    assert near_zero < at_peak


# ---------------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------------

def _one_channel_group_synth(values):
    return {"emissionNames": ["e"], "injectionLocation": 0.0, "injectionLocation_rad": 0.0,
            "e": {"data": [[[v] for v in values]], "observed_phi_loc": [[[0.0] for _ in values]], "scaleSynth": 1.0}}


def test_residual_is_signed_and_leaves_out_bad_channels():
    p = Parameters()
    p.add("a_0", value=1.0)
    p.add("b_e_0", value=0.0)
    synth = _one_channel_group_synth([1.0, 1.0, 1.0, 1.0])
    data = {"observed": [[2.0, 0.5, -0.1, 7.0]], "observed_error": [[1.0, 1.0, 1.0, np.inf]]}
    res = U.residual(p, data, synth, "constant", True, 0.0)
    assert res == pytest.approx([1.0, -0.5, -1.1])            # 3 channels: the bad one is gone


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------

BASE = {"BOLOMETERS": {}, "shot": 1, "radDistDirectories_LocIndependent": ["x"]}


@pytest.mark.parametrize("change, match", [
    ({"fit_method": "leastsq"}, "fit_method"),
    ({"error_floor": -0.1}, ">= 0"),
    ({"error_floor": 0.0}, "near-zero"),
    ({"noise_window": [5.0, 1.0]}, "noise_window"),
    ({"BOLOMETERS": {"A": {"badChannels": "A01"}}}, "badChannels"),
])
def test_run_config_checks(change, match):
    with pytest.raises(ValueError, match=match):
        RunConfig.from_dict({**BASE, **change})


def test_zero_floor_is_fine_with_a_noise_window():
    assert RunConfig.from_dict({**BASE, "error_floor": 0.0, "noise_window": [0.0, 1.0]}).error_floor == 0.0


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

class _Fit:
    def __init__(self, chisqr, ndata, nvarys):
        self.aic = ndata * np.log(chisqr / ndata) + 2 * nvarys


def test_score_is_aic_and_charges_for_parameters():
    simple, flexible = _Fit(10.0, 40, 1), _Fit(10.0, 40, 3)
    assert U.fit_score(simple) < U.fit_score(flexible)
    assert U.fit_score(object()) == np.inf


# ---------------------------------------------------------------------------
# Through Emis3D on fabricated MOCK data
# ---------------------------------------------------------------------------

def _zero_channel(root, cam, index):
    with h5py.File(root / "measurements" / "MOCK" / "1" / f"{cam}.h5", "r+") as f:
        f["DATA_CALIBRATED"][index, :] = 0.0


def test_listed_and_all_zero_channels_are_left_out(tmp_path, caplog):
    tags, _ = cam_info("POL_A")
    run = write_workspace(tmp_path, BOLOMETERS={c: {"dataFileName": f"1/{c}.h5", **({"badChannels": [tags[2]]} if c == "POL_A" else {})} for c in CAMS}, **FIT)
    _zero_channel(tmp_path, "POL_B", 4)
    e = Emis3D(runConfigName=str(run))
    e._prepare_fits(evalTime=10.0)
    err = {c: np.asarray(e.fitData[10.0]["boloData_error"][c]) for c in ("POL_A", "POL_B", "POL_C")}
    assert np.isinf(err["POL_A"][2]) and np.isinf(err["POL_B"][4])
    assert np.all(np.isfinite(err["POL_C"]))
    assert "zero for the whole shot" in caplog.text


def test_unknown_bad_channel_is_an_error(tmp_path):
    run = write_workspace(tmp_path, BOLOMETERS={c: {"dataFileName": f"1/{c}.h5", "badChannels": ["NOPE"] if c == "POL_A" else []} for c in CAMS}, **FIT)
    with pytest.raises(ValueError, match="not in its data file"):
        Emis3D(runConfigName=str(run))


def test_noise_window_sets_the_floor(tmp_path):
    """With 1% noise and dt averaging, the measured floor replaces a tiny configured one."""
    run = write_workspace(tmp_path, **{**FIT, "error_floor": 1e-9, "error_relative": 0.0, "noise_window": [9.0, 9.5]})
    e = Emis3D(runConfigName=str(run))
    e._prepare_fits(evalTime=10.0)
    err = np.asarray(e.fitData[10.0]["boloData_error"]["POL_C"])
    noise = e._channel_noise("POL_C", 10.0)
    assert err == pytest.approx(noise, rel=1e-9) and np.all(err > 1e-6)


@pytest.mark.parametrize("method", ["least_squares", "differential_evolution"])
def test_fit_picks_the_true_raddist(tmp_path, method):
    run = write_workspace(tmp_path, miscal={c: 1.0 for c in CAMS}, **{**FIT, "fit_method": method})
    e = Emis3D(runConfigName=str(run))
    e._prepare_fits(evalTime=10.0)
    e._minimize_radDists(evalTime=10.0)
    best = int(np.argmin(e.fits[10.0]["scores"]))
    number = e.fits[10.0][best]["info"]["radDists"]["radDistNumber"]
    label = e.data["synthetic"]["locIndependent"][number].info["radDistPath"]
    assert label.endswith(f"R_{TRUE_RD}.json")
    assert np.all(np.isfinite(e.fits[10.0]["scores"]))


def test_class_and_script_paths_rank_identically(tmp_path):
    run = write_workspace(tmp_path, miscal={c: 1.0 for c in CAMS}, **FIT)
    e = Emis3D(runConfigName=str(run))
    e._prepare_fits(evalTime=10.0)
    e._minimize_radDists(evalTime=10.0)
    class_scores = np.array(e.fits[10.0]["scores"], dtype=float)
    script_scores = []
    for ii in sorted(k for k in e.fits[10.0] if isinstance(k, int)):
        job = (ii, e.fits[10.0][ii]["parameters"], e.fitData[10.0], e.fits[10.0][ii]["synthetic_dict"],
               e.info["scale_def"], e.info["helical_endpoint_weight"], e.info["fit_method"])
        script_scores.append(U.fit_score(U.runParallel(job)[1]))
    assert np.allclose(class_scores, script_scores)


def test_every_fit_failing_is_a_clear_error(tmp_path):
    run = write_workspace(tmp_path, **FIT)
    e = Emis3D(runConfigName=str(run))
    e._prepare_fits(evalTime=10.0)
    e.fits[10.0]["scores"][:] = np.inf
    with pytest.raises(RuntimeError, match="Every fit failed"):
        e._post_process_fit_arrangement(evalTime=10.0)
