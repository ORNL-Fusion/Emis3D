"""
tests/test_util_emis3d.py
=========================
Unit tests for src/emis3d/Util_emis3D.py
Covers find_dphi, all scale_* functions, error models,
and the scale_wrapper dispatcher.
No cherab or raysect required.
"""

import numpy as np

from emis3d.util_emis3D import (
    find_dphi,
    scale_exp,
    scale_linear,
    scale_constant,
    scale_wrapper,
    error_exponential,
    error_inverse,
    error_inv_sqrt,
)

TWO_PI = 2 * np.pi

# ---------------------------------------------------------------------------
# find_dphi
# ---------------------------------------------------------------------------


class TestFindDphi:

    def test_zero_offset(self):
        """phi == mu should give dphi == 0."""
        phi = np.array([1.0])
        result = find_dphi(phi, mu=1.0)
        assert np.allclose(result[0], 0.0)

    def test_clockwise_direction(self):
        """Clockwise emission: result should be negative for phi behind mu."""
        phi = np.array([np.deg2rad(60)])
        result = find_dphi(phi, mu=np.deg2rad(30), emissionName="clockwise")
        # cw = (2 - 1) % 2pi = 1; result = -1
        assert np.isclose(result[0], np.deg2rad(330))

    def test_counterclock_direction(self):
        """CounterClock emission: result should be positive for phi ahead of mu."""
        phi = np.array([np.deg2rad(60)])
        result = find_dphi(phi, mu=np.deg2rad(30), emissionName="counterClock")
        # ccw = (2 - 1) % 2pi = 1; result = +1
        assert np.isclose(result[0], np.deg2rad(30))

    def test_angles_are_physical(self):
        """phi and phi + 2 pi are the same place: the distance does not change."""
        for name in ("", "counterClock_rev0", "clockwise_rev0"):
            d1 = find_dphi(np.array([0.5]), mu=0.0, emissionName=name)
            d2 = find_dphi(np.array([0.5 + 2 * np.pi]), mu=0.0, emissionName=name)
            d3 = find_dphi(np.array([0.5 - 4 * np.pi]), mu=0.0, emissionName=name)
            assert np.allclose(d1, d2) and np.allclose(d1, d3), name

    def test_revolution_comes_from_the_emission_name(self):
        phi, mu = np.array([np.deg2rad(275.0)]), np.deg2rad(260.0)
        for k in range(4):
            ccw = find_dphi(phi, mu, emissionName=f"counterClock_rev{k}")[0]
            cw = find_dphi(phi, mu, emissionName=f"clockwise_rev{k}")[0]
            assert np.isclose(np.rad2deg(ccw), 15.0 + 360.0 * k)
            assert np.isclose(np.rad2deg(cw), 345.0 + 360.0 * k)

    def test_returns_ndarray(self):
        result = find_dphi(np.array([1.0, 2.0]), mu=0.5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)


# ---------------------------------------------------------------------------
# scale_exp
# ---------------------------------------------------------------------------


class TestScaleExp:

    def test_peak_at_zero(self):
        """scale_exp(A, B, 0) should equal A."""
        assert np.isclose(scale_exp(3.0, 2.0, np.array([0.0]))[0], 3.0)

    def test_decays_with_distance(self):
        """Larger |dphi| → smaller value for B > 0."""
        v0 = scale_exp(1.0, 1.0, np.array([0.0]))[0]
        v1 = scale_exp(1.0, 1.0, np.array([1.0]))[0]
        v2 = scale_exp(1.0, 1.0, np.array([2.0]))[0]
        assert v0 > v1 > v2

    def test_b_zero_is_constant(self):
        """B=0 → exp(0) = 1 → constant A."""
        dphi = np.array([0.0, 1.0, 2.0])
        result = scale_exp(5.0, 0.0, dphi)
        assert np.allclose(result, 5.0)

    def test_symmetry(self):
        """scale_exp should be symmetric: dphi and -dphi give the same value."""
        dphi_pos = np.array([1.5])
        dphi_neg = np.array([-1.5])
        assert np.isclose(scale_exp(1.0, 2.0, dphi_pos)[0], scale_exp(1.0, 2.0, dphi_neg)[0])


# ---------------------------------------------------------------------------
# scale_linear
# ---------------------------------------------------------------------------


class TestScaleLinear:

    def test_gradient(self):
        dphi = np.array([0.0, 1.0, 2.0])
        result = scale_linear(2.0, 1.0, dphi)  # y = 2*dphi + 1
        assert np.allclose(result, [2.0, 1.0, 0.0])

    def test_stays_at_zero_past_the_end(self):
        """It used to take abs(), so the profile rose again: 2 - 1 * 5 gave 3."""
        result = scale_linear(2.0, 1.0, np.array([2.0, 3.0, 5.0, 50.0]))
        assert np.allclose(result, 0.0)

    def test_sign_of_slope_does_not_matter(self):
        assert np.allclose(scale_linear(2.0, -1.0, np.array([0.5, 4.0])), [1.5, 0.0])


# ---------------------------------------------------------------------------
# scale_constant
# ---------------------------------------------------------------------------


class TestScaleConstant:

    def test_returns_uniform_A(self):
        dphi = np.array([0.0, 1.0, -2.0, 5.0])
        result = scale_constant(4.2, dphi)
        assert np.allclose(result, 4.2)

    def test_shape_preserved(self):
        dphi = np.zeros(7)
        assert scale_constant(1.0, dphi).shape == (7,)


# ---------------------------------------------------------------------------
# scale_wrapper  (dispatcher)
# ---------------------------------------------------------------------------


class TestScaleWrapper:

    def _dphi(self):
        return np.linspace(-np.pi, np.pi, 50)

    def test_exponential_dispatch(self):
        dphi = np.array([0.0, 0.5, 1.0, -0.5, -1.0])
        result = scale_wrapper(2.0, 1.0, phi=dphi, scale_def="exponential", dphi=dphi)
        # Peak should be at dphi=0
        assert np.isclose(result[0], 2.0, atol=1e-12)

    def test_linear_dispatch(self):
        dphi = np.array([0.0, 1.0, 2.0])
        result = scale_wrapper(3.0, 1.0, phi=dphi, scale_def="linear", dphi=dphi)
        assert np.allclose(result, scale_linear(3.0, 1.0, dphi))

    def test_constant_dispatch(self):
        dphi = np.ones(10)
        result = scale_wrapper(5.0, 0.0, phi=dphi, scale_def="constant", dphi=dphi)
        assert np.allclose(result, 5.0)

    def test_unknown_scale_def_returns_ones(self):
        """Unknown scale_def should fall back to an array of 1s."""
        dphi = np.array([1.0, 2.0, 3.0])
        result = scale_wrapper(99.0, 99.0, phi=dphi, scale_def="bogus_def", dphi=dphi)
        assert np.allclose(result, 1.0)

    def test_dphi_computed_when_none(self):
        """If dphi is None it should be derived from phi and mu."""
        phi = np.array([0.5, 1.0, 1.5])
        mu = 1.0
        result_from_dphi = scale_wrapper(1.0, 0.5, phi=phi, mu=mu, scale_def="exponential", dphi=None)
        dphi_manual = find_dphi(phi, mu)
        result_manual = scale_exp(1.0, 0.5, dphi_manual)
        assert np.allclose(result_from_dphi, result_manual)


# ---------------------------------------------------------------------------
# Error models
# ---------------------------------------------------------------------------


class TestErrorModels:

    def test_error_exponential_at_zero(self):
        """At signal=0 the error should equal scale_factor."""
        err = error_exponential(np.array([0.0]), max_signal=1.0, scale_factor=2.0)
        assert np.isclose(err[0], 2.0)

    def test_error_exponential_decreasing(self):
        s = np.array([0.0, 0.25, 0.5, 1.0])
        err = error_exponential(s, max_signal=1.0)
        assert np.all(np.diff(err) < 0)

    def test_error_exponential_clips_negative(self):
        """Negative signal should be treated as 0 (no negative error)."""
        err = error_exponential(np.array([-1.0]), max_signal=1.0)
        err_zero = error_exponential(np.array([0.0]), max_signal=1.0)
        assert np.isclose(err[0], err_zero[0])

    def test_error_inverse_at_peak(self):
        """At signal=max_signal, error = scale_factor."""
        err = error_inverse(np.array([1.0]), max_signal=1.0, scale_factor=3.0)
        assert np.isclose(err[0], 3.0)

    def test_error_inverse_increasing(self):
        """Larger signal → smaller error (inverse relationship)."""
        s = np.array([0.5, 1.0, 2.0])
        err = error_inverse(s, max_signal=2.0)
        assert np.all(np.diff(err) < 0)

    def test_error_inv_sqrt_at_peak(self):
        """At signal = max_signal, error = scale_factor * sqrt(1) = scale_factor."""
        err = error_inv_sqrt(np.array([5.0]), max_signal=5.0, scale_factor=2.0)
        assert np.isclose(err[0], 2.0)

    def test_error_inv_sqrt_positive(self):
        """Errors should always be positive."""
        s = np.array([0.1, 1.0, 10.0])
        assert np.all(error_inv_sqrt(s, max_signal=10.0) > 0)

    def test_all_error_models_return_ndarray(self):
        s = np.array([0.5, 1.0])
        for fn in (error_exponential, error_inverse, error_inv_sqrt):
            result = fn(s, max_signal=1.0)
            assert isinstance(result, np.ndarray), f"{fn.__name__} should return ndarray"
