"""
tests/test_step_scale_library.py

scale_def: step with the library solver. The step scale has one amplitude
per camera toroidal location (a_<tag> at the first, a_<tag>_step<k> at the
others, in sorted order), and every toroidal angle takes the amplitude of
its nearest location.

(The library solver used to create a single amplitude and read the second
from a decay parameter that the step scale no longer has, so every other
location was fitted with an amplitude of exactly zero.)
"""

import numpy as np
import pytest

import emis3d.uncertainty as U
from emis3d import util_emis3D as Util_emis3D
from emis3d.emis3D import Emis3D
from test_calibration import TRUE_RD, write_workspace

# One factor per toroidal location: WRAP 20, POL_A 45, POL_B and SKEW 165, POL_C 285 deg
FACTORS = {"WRAP": 0.9, "POL_A": 1.0, "POL_B": 0.6, "SKEW": 0.6, "POL_C": 1.4}
LOCATIONS_DEG = [20.0, 45.0, 165.0, 285.0]


def test_scale_step_takes_the_nearest_location_around_the_torus():
    locs = np.deg2rad([20.0, 165.0, 285.0])
    A = np.array([1.0, 2.0, 3.0])
    phi = np.deg2rad([20.0, 90.0, 100.0, 165.0, 230.0, 285.0, 330.0, 340.0, 359.0, 0.0])
    #                 20    20    165    165    285    285    285    20     20     20   (330 is 45 from 285, 50 from 20)
    np.testing.assert_array_equal(Util_emis3D.scale_step(A, phi, locs), [1, 1, 2, 2, 3, 3, 3, 1, 1, 1])
    # half way between two locations: one value per angle, the first location's
    np.testing.assert_array_equal(Util_emis3D.scale_step(A, np.deg2rad([92.5]), locs), [1.0])
    with pytest.raises(ValueError, match="one amplitude per bolometer location"):
        Util_emis3D.scale_step(np.array([1.0, 2.0]), phi, locs)


def test_step_parameter_names_and_locations():
    assert Util_emis3D.step_param_name("225", 0) == "a_225" and Util_emis3D.step_param_name("225", 2) == "a_225_step2"
    np.testing.assert_allclose(Util_emis3D.unique_locations(np.deg2rad([285.0, 20.0, 380.0, 165.0, 165.0])), np.deg2rad([20.0, 165.0, 285.0]))


def fitted(tmp_path):
    run = write_workspace(tmp_path, tstart=10.0, miscal=FACTORS, scale_def="step")
    e = Emis3D(runConfigName=str(run))
    e._prepare_fits(evalTime=10.0)
    e._minimize_radDists(evalTime=10.0)
    best = e.fits[10.0][int(np.argmin(e.fits[10.0]["scores"]))]
    return e, best


def test_library_fit_has_one_amplitude_per_camera_location(tmp_path):
    e, best = fitted(tmp_path)
    np.testing.assert_allclose(np.rad2deg(e._step_locations()), LOCATIONS_DEG)
    np.testing.assert_allclose(np.rad2deg(best["synthetic_dict"]["step_locations"]), LOCATIONS_DEG)
    names = [n for n in best["parameters"] if n.startswith("a_")]
    assert names == ["a_0", "a_0_step1", "a_0_step2", "a_0_step3"]            # five cameras at four locations
    assert all(best["parameters"][n].vary for n in names)
    assert not any(n.startswith("b_") for n in best["parameters"])             # the step has no decay


def test_library_fit_recovers_every_locations_amplitude(tmp_path):
    e, best = fitted(tmp_path)
    assert best["info"]["radDists"]["radDistNumber"] == _number_of(e, f"R_{TRUE_RD}.json")
    p = best["fit"].params.valuesdict()
    amps = Util_emis3D.step_amplitudes(p, "0", 4)
    truth = np.array([FACTORS["WRAP"], FACTORS["POL_A"], FACTORS["POL_B"], FACTORS["POL_C"]])
    np.testing.assert_allclose(amps / amps[1], truth / truth[1], rtol=0.02)

    # every camera is fitted (the second, fourth, ... locations used to get a model of exactly zero)
    model = Util_emis3D.residual(best["fit"].params, None, best["synthetic_dict"], "step", residual=False)["elongatedRing"]
    for g, cam in enumerate(e.channel_order["bolometer_order"]):
        data = np.array(e.fitData[10.0]["observed"][g])
        assert np.max(model[g]) == pytest.approx(data.max(), rel=0.05), cam


def test_toroidal_profile_uses_the_fitted_steps(tmp_path):
    e, best = fitted(tmp_path)
    p = best["fit"].params.valuesdict()
    synth = best["synthetic_dict"]
    prof = U.toroidal_profile(p, synth, "step")
    phi, P = np.rad2deg(prof["phi"]), prof["total_power"]
    amps = Util_emis3D.step_amplitudes(p, "0", 4)
    scale = float(synth["elongatedRing"]["scaleSynth"])                       # P_pol is 1 everywhere in these radDists
    for loc, amp in zip(LOCATIONS_DEG, amps, strict=True):
        assert P[np.argmin(np.abs(phi - loc))] == pytest.approx(amp * scale)
    q, _ = U.derived_quantities(p, synth, "step")
    assert q["tpf"] == pytest.approx(P.max() / P.mean())


def test_default_locations_are_the_angles_the_raddist_was_observed_at():
    """Without camera locations (a radDist used on its own), the distinct observed angles are the locations."""
    from emis3d.radDistFitting import RadDistFitting

    rd = RadDistFitting()
    rd.info.update({"units": "Brightness", "emissionNames": ["ring"], "injectionLocation": 225.0})
    rd.data = {"Brightness": {"channelOrder": {"A": ["A1", "A2"], "B": ["B1"]}, "ring": {"A": [[1.0], [2.0]], "B": [[3.0]]}},
               "Brightness_error": {"ring": {"A": [[0.0], [0.0]], "B": [[0.0]]}},
               "observed_phi_loc": {"ring": {"A": [[4.0], [4.0]], "B": [[1.0]]}}}
    rd._map_signals()
    rd.prepare_for_fits([["A1", "A2"], ["B1"]])
    rd.create_parameters(scale_def="step")
    assert rd.fitSynthetic["step_locations"] == [1.0, 4.0]
    assert rd.fitSynthetic["params"]["paramName"] == ["a_225", "a_225_step1"]
    rd.create_parameters(scale_def="constant")                                # no step: no locations left behind
    assert "step_locations" not in rd.fitSynthetic


def _number_of(e, file_name):
    for number, rd in e.data["synthetic"]["locIndependent"].items():
        if str(rd.info["radDistPath"]).endswith(file_name):
            return number
    raise AssertionError(f"{file_name} not loaded")


def test_cameras_of_the_fit_only(tmp_path):
    """Only cameras with a channel in the fit give a location."""
    run = write_workspace(tmp_path, tstart=10.0, miscal=FACTORS, scale_def="step",
                          BOLOMETERS={c: {"dataFileName": f"1/{c}.h5"} for c in ("POL_A", "POL_C")})
    e = Emis3D(runConfigName=str(run))
    np.testing.assert_allclose(np.rad2deg(e._step_locations()), [45.0, 285.0])
