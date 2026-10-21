"""
tests/test_results.py

Post-processing through Emis3D on fabricated MOCK data (no ray tracing):
the toroidal profile and TPF (review issue 5), the uncertainty ranges
(Phase 6), the camera locations used by the step scale, and the TPF panel
(S1).
"""

import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from emis3d.emis3D import Emis3D
from emis3d.tokamak import Tokamak
from test_calibration import CAMS, write_workspace

T = 10.0
FIT = {"error_definition": "floor_relative", "scale_def": "exponential", "enable_dphi_scaling": True,
       "uncertainty": {"n_samples": 300, "max_candidates": 3}}


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    run = write_workspace(tmp_path_factory.mktemp("ws"), miscal={c: 1.0 for c in CAMS}, **FIT)
    e = Emis3D(runConfigName=str(run))
    e._prepare_fits(evalTime=T)
    e._minimize_radDists(evalTime=T)
    best = int(np.argmin(e.fits[T]["scores"]))
    # what _post_process_fit_arrangement stores, without rebuilding the (fabricated) radDist
    e.bestFits = {T: {"fit": e.fits[T][best]["fit"], "synthetic_dict": e.fits[T][best]["synthetic_dict"]}}
    e._post_process_calculations(evalTime=T)
    e._post_process_uncertainty(evalTime=T)
    return e


def test_profile_keeps_every_sample_and_tpf_is_peak_over_mean(fitted):
    rad = fitted.bestFits[T]["radiation_distribution"]
    assert len(rad["phi"]) == len(rad["total_power"]) == 100          # no sample deleted
    P = np.asarray(rad["total_power"])
    assert rad["toroidal_peaking_factor"] == pytest.approx(P.max() / P.mean())


def test_ranges_are_stored_and_contain_the_best_fit(fitted):
    ranges = fitted.bestFits[T]["ranges"]
    tpf = fitted.bestFits[T]["radiation_distribution"]["toroidal_peaking_factor"]
    lo, hi = ranges["total"]["tpf"]["95%"]
    assert lo <= tpf <= hi
    for q in ("total_power", "peak_phi_deg", "peak_power", "amplitude"):
        assert q in ranges["total"]
    assert set(ranges["profile_band"]) >= {"phi", "median", "68%", "95%"}


def test_ranges_can_be_switched_off(tmp_path):
    run = write_workspace(tmp_path, miscal={c: 1.0 for c in CAMS}, **{**FIT, "uncertainty": {"enabled": False}})
    e = Emis3D(runConfigName=str(run))
    e._prepare_fits(evalTime=T)
    e._minimize_radDists(evalTime=T)
    e.bestFits = {T: {}}
    e._post_process_uncertainty(evalTime=T)
    assert "ranges" not in e.bestFits[T]


def test_camera_locations_are_found_by_channel_name(fitted):
    """This compared camera names with channel names, so it always found none."""
    tok = Tokamak(tokamakName="MOCK", mode="Analysis", loadBolometers=True, workspace=fitted.ws)
    fitted.bestFits[T]["radDist"] = types.SimpleNamespace(tokamak=tok)
    fitted._post_process_bolo_locations(evalTime=T)
    found = sorted(np.rad2deg(fitted.bestFits[T]["bolometer_phi_locations"]).round(6))
    assert found == [20.0, 45.0, 165.0, 285.0]                    # SKEW shares 165 with POL_B


def test_tpf_panel_draws_the_band_and_a_full_height_peak_marker(fitted):
    f, ax = plt.subplots()
    try:
        fitted._plot_tpf_panel(ax, T)
        marker = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"][0]
        assert list(marker.get_ydata()) == [0, 1]                  # axes fraction: bottom to top
        assert "68%" in ax.get_title()
        assert len(ax.collections) == 2                            # 68% and 95% bands
    finally:
        plt.close(f)
