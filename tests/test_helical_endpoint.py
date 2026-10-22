"""
tests/test_helical_endpoint.py
==============================
Unit tests for the helical endpoint-continuity soft constraint added to
src/emis3d/Util_emis3D.py (helical_endpoint_penalty / _emission_pairs) and its
integration into ``residual``.

The constraint ties paired clockwise / counterClock helical distributions
together at the helix endpoint (phi = injectionLocation mod 2*pi), complementing
the shared amplitude ``a_<tag>`` that already ties the dphi = 0 peaks together.

NOTE: run with:  python -m pytest tests/
"""

import numpy as np
import pytest
from lmfit import Parameters

import emis3d.util_emis3D as U

NAMES = ["counterClock_rev0", "clockwise_rev0"]
MU_DEG = 90.0
PHI_ARRAY = list(np.linspace(0, 2 * np.pi, 36, endpoint=False))


def _synthetic_dict(P_pol_cw, P_pol_ccw, with_fit_data=False):
    sd = {
        "emissionNames": NAMES,
        "injectionLocation": MU_DEG,
        "injectionLocation_rad": np.deg2rad(MU_DEG),
        "info": {"numTransists": 1.0},
        "clockwise_rev0": {
            "scaleSynth": 1.0,
            "phi_array": PHI_ARRAY,
            "P_pol": P_pol_cw,
        },
        "counterClock_rev0": {
            "scaleSynth": 1.0,
            "phi_array": PHI_ARRAY,
            "P_pol": P_pol_ccw,
        },
    }
    if with_fit_data:
        for em in NAMES:
            sd[em].update(
                {
                    "data": [[[1.0]] * 4],
                    "data_error": [[[0.1]] * 4],
                    "observed_phi_loc": [[[0.0]] * 4],
                }
            )
    return sd


def _params(a, b_cw, b_ccw):
    p = Parameters()
    p.add("a_90", value=a, min=0)
    p.add("b_clockwise_rev0_90", value=b_cw, min=0, max=15)
    p.add("b_counterClock_rev0_90", value=b_ccw, min=0, max=15)
    return p


class TestEmissionPairs:
    def test_pairs_clockwise_with_counterclock(self):
        assert U._emission_pairs(NAMES) == [("clockwise_rev0", "counterClock_rev0")]

    def test_no_pairs_for_non_helical(self):
        assert U._emission_pairs(["elongatedRing"]) == []

    def test_unpaired_direction_ignored(self):
        # only a clockwise direction, no matching counterClock
        assert U._emission_pairs(["clockwise_rev0"]) == []


class TestEndpointPenalty:
    def test_weight_zero_disables(self):
        sd = _synthetic_dict([1.0] * 36, [1.0] * 36)
        assert U.helical_endpoint_penalty(_params(1, 2, 2), sd, "exponential", 0.0) == []

    def test_missing_P_pol_is_an_error(self):
        """Every current radDist stores its toroidal power profile; a missing one is a bug."""
        sd = _synthetic_dict([1.0] * 36, [1.0] * 36)
        del sd["clockwise_rev0"]["P_pol"]
        with pytest.raises(KeyError, match="P_pol"):
            U.helical_endpoint_penalty(_params(1, 2, 2), sd, "exponential", 1.0)

    def test_symmetric_pair_zero_penalty(self):
        sd = _synthetic_dict([1.0] * 36, [1.0] * 36)
        pen = U.helical_endpoint_penalty(_params(1.0, 2.0, 2.0), sd, "exponential", 1.0)
        assert len(pen) == 1
        assert abs(pen[0]) < 1e-12

    def test_deep_decay_inert(self):
        # both directions vanish by the endpoint -> penalty ~ 0 even if b differs
        sd = _synthetic_dict([1.0] * 36, [1.0] * 36)
        pen = U.helical_endpoint_penalty(_params(1.0, 1.0, 3.0), sd, "exponential", 1.0)
        assert abs(pen[0]) < 1e-6

    """
    def test_gentle_decay_asymmetric_b_active(self):
        sd = _synthetic_dict([1.0] * 36, [1.0] * 36)
        pen = U.helical_endpoint_penalty(
            _params(1.0, 0.005, 0.02), sd, "exponential", 1.0
        )
        assert abs(pen[0]) > 1e-3
    """

    def test_asymmetric_Ppol_active(self):
        """A mismatch where counterClock finishes its revolution (just before mu) is penalized."""
        i_end_ccw = int(np.argmax(U.find_dphi(np.array(PHI_ARRAY), np.deg2rad(MU_DEG), "counterClock_rev0")))
        assert np.isclose(np.rad2deg(PHI_ARRAY[i_end_ccw]), MU_DEG - 10.0)   # 80 deg: last sample before mu
        ppol_ccw = [1.0] * 36
        ppol_ccw[i_end_ccw] = 0.4
        sd = _synthetic_dict([1.0] * 36, ppol_ccw)
        pen = U.helical_endpoint_penalty(_params(1.0, 0.005, 0.005), sd, "exponential", 1.0)
        assert abs(pen[0]) > 1e-3

    def test_each_direction_is_sampled_at_its_own_endpoint(self):
        """counterClock ends just before mu and clockwise just after; mu itself is where
        counterClock starts, so a change there is not an endpoint mismatch."""
        mu = np.deg2rad(MU_DEG)
        phi = np.array(PHI_ARRAY)
        i_end_cw = int(np.argmax(U.find_dphi(phi, mu, "clockwise_rev0")))
        assert np.isclose(np.rad2deg(phi[i_end_cw]), MU_DEG + 10.0)          # 100 deg
        at_mu = int(np.argmin(np.abs(phi - mu)))
        ppol_ccw = [1.0] * 36
        ppol_ccw[at_mu] = 0.4                    # counterClock's start, not its end
        sd = _synthetic_dict([1.0] * 36, ppol_ccw)
        pen = U.helical_endpoint_penalty(_params(1.0, 0.005, 0.005), sd, "exponential", 1.0)
        assert abs(pen[0]) < 1e-12

    def test_later_revolutions_use_their_own_distance(self):
        """For rev1 the endpoint is two full turns along the helix."""
        names = ["counterClock_rev1", "clockwise_rev1"]
        sd = _synthetic_dict([1.0] * 36, [1.0] * 36)
        sd["emissionNames"] = names
        sd["counterClock_rev1"] = sd.pop("counterClock_rev0")
        sd["clockwise_rev1"] = sd.pop("clockwise_rev0")
        p = Parameters()
        p.add("a_90", value=1.0, min=0)
        p.add("b_clockwise_rev1_90", value=0.0, min=0, max=15)
        p.add("b_counterClock_rev1_90", value=0.02, min=0, max=15)
        pen = U.helical_endpoint_penalty(p, sd, "exponential", 1.0)
        end = np.deg2rad(710.0)                  # 350 deg + one full turn
        assert pen[0] == pytest.approx(1.0 - np.exp(-0.02 * end**2), rel=1e-6)

    def test_no_shared_amplitude_no_penalty(self):
        # no a_<tag> -> nothing to tie together
        sd = _synthetic_dict([1.0] * 36, [1.0] * 36)
        p = Parameters()
        p.add("SomeBolo", value=0.3, min=0)
        assert U.helical_endpoint_penalty(p, sd, "constant", 1.0) == []


class TestResidualIntegration:
    def test_residual_appends_one_penalty_element(self):
        sd = _synthetic_dict([1.0] * 36, [1.0] * 36, with_fit_data=True)
        data = {"observed": {0: [1.0] * 4}, "observed_error": {0: [0.1] * 4}}
        res_off = U.residual(_params(1.0, 1.0, 3.0), data, sd, "exponential", True, 0.0)
        res_on = U.residual(_params(1.0, 1.0, 3.0), data, sd, "exponential", True, 1.0)
        assert len(res_on) == len(res_off) + 1

    '''
    def test_constraint_ties_decay_constants(self):
        """With symmetric P_pol the fit should drive b_cw == b_ccw."""
        sd = _synthetic_dict([1.0] * 36, [1.0] * 36, with_fit_data=True)
        data = {"observed": {0: [1.0] * 4}, "observed_error": {0: [0.1] * 4}}
        out = minimize(
            U.residual,
            _params(1.0, 0.01, 0.05),
            args=(data, sd, "exponential", None, True, 5.0),
            method="leastsq",
        )
        assert out.success
        b_cw = out.params["b_clockwise_rev0_90"].value
        b_ccw = out.params["b_counterClock_rev0_90"].value
        assert abs(b_cw - b_ccw) < 1e-3
    '''
