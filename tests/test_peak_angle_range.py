"""
tests/test_peak_angle_range.py

The range of the peak's toroidal angle (peak_phi_deg) around 0/360 deg.
Draws of 359.9 and 0.1 deg are 0.2 deg apart, but their percentiles were
taken as plain numbers: with the injector at 0 deg (the default
injectionLocation) the 95% range of the peak location came out as (0, 360).
Angles are now unwrapped around the best fit's value first.
"""

import numpy as np
import pytest

from emis3d import uncertainty as U


def result(peak_deg, best_peak, tpf=None):
    n = len(peak_deg)
    samples = {"peak_phi_deg": np.asarray(peak_deg, dtype=float), "tpf": np.full(n, 1.5) if tpf is None else np.asarray(tpf, dtype=float)}
    best = {"tpf": 1.5}
    if best_peak is not None:
        best["peak_phi_deg"] = best_peak
    return U.UncertaintyResult(samples=samples, source=np.ones(n, dtype=int), candidate=np.zeros(n, dtype=int), phi=np.linspace(0, 2 * np.pi, 4, endpoint=False),
                               profiles=np.ones((n, 4)), weights=np.array([1.0]), candidate_ids=[0], best_score=0.0, flags={},
                               config=U.UncertaintyConfig(), best_values=best)


def spread(centre, width=0.5, n=4001, seed=0):
    """Draws around an angle, reported in [0, 360) as derived_quantities reports them."""
    return (centre + np.random.default_rng(seed).normal(0.0, width, n)) % 360.0


@pytest.mark.parametrize("centre", [0.0, 359.8, 0.3, 360.0])
def test_range_of_a_peak_at_the_seam_is_narrow(centre):
    s = result(spread(centre), best_peak=centre % 360.0).summary("total")["peak_phi_deg"]
    for level, half in (("68%", 0.5), ("95%", 1.0)):
        lo, hi = s[level]
        assert lo <= hi
        assert hi - lo == pytest.approx(2 * half, rel=0.15)                    # about +- 1 (2) sigma, not the whole circle
        assert lo <= centre % 360.0 <= hi or lo <= centre % 360.0 - 360.0 <= hi  # around the best value, continuously
    assert min(s["median"], 360.0 - s["median"]) == pytest.approx(min(centre % 360.0, 360.0 - centre % 360.0), abs=0.05)
    assert 0.0 <= s["median"] < 360.0


def test_range_runs_continuously_through_zero():
    lo, hi = result(spread(0.0), best_peak=0.0).summary("total")["peak_phi_deg"]["95%"]
    assert lo == pytest.approx(-1.0, abs=0.1) and hi == pytest.approx(1.0, abs=0.1)
    lo, hi = result(spread(359.9), best_peak=359.9).summary("total")["peak_phi_deg"]["95%"]
    assert lo == pytest.approx(358.9, abs=0.1) and hi == pytest.approx(360.9, abs=0.1)


def test_a_peak_away_from_the_seam_is_unchanged():
    draws = spread(225.0)
    s = result(draws, best_peak=225.0).summary("total")["peak_phi_deg"]
    assert s["median"] == pytest.approx(np.median(draws))
    assert s["95%"] == pytest.approx(tuple(np.percentile(draws, [2.5, 97.5])))


def test_without_a_best_value_the_draws_circular_mean_is_the_centre():
    s = result(spread(0.0), best_peak=None).summary("total")["peak_phi_deg"]
    lo, hi = s["95%"]
    assert hi - lo == pytest.approx(2.0, rel=0.15) and lo < 0.0 < hi


def test_other_quantities_are_not_unwrapped():
    tpf = np.linspace(1.0, 400.0, 4001)                                       # plain numbers, whatever their size
    s = result(spread(0.0), best_peak=0.0, tpf=tpf).summary("total")["tpf"]
    assert s["median"] == pytest.approx(np.median(tpf)) and s["95%"][1] == pytest.approx(np.percentile(tpf, 97.5))
