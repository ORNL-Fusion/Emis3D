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


# ---------------------------------------------------------------------------
# Best-fit figure
# ---------------------------------------------------------------------------

def test_tpf_panel_is_centred_on_the_injector(fitted):
    f, ax = plt.subplots()
    try:
        fitted._plot_tpf_panel(ax, T)
        best_fit = [ln for ln in ax.get_lines() if ln.get_label() == "best fit"][0]
        x = np.asarray(best_fit.get_xdata())
        assert x.min() >= -180 and x.max() < 180 and np.all(np.diff(x) > 0)
        peak = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"][0]
        assert abs(peak.get_xdata()[0]) < 1e-9                       # the peak is at the injector
        assert "total power" in ax.get_title()
    finally:
        plt.close(f)


def test_cameras_at_one_angle_share_a_marker(fitted):
    tok = Tokamak(tokamakName="MOCK", mode="Analysis", loadBolometers=True, workspace=fitted.ws)
    f, ax = plt.subplots()
    try:
        fitted._plot_tpf_panel(ax, T, tok=tok)
        labels = sorted(t.get_text().strip() for t in ax.texts if t.get_text().strip() != "peak")
        assert labels == ["POL_A", "POL_B, SKEW", "POL_C", "WRAP"]
    finally:
        plt.close(f)


def test_signal_panel_residuals_and_not_fitted_channels(fitted):
    cam = "POL_C"
    err = fitted.fitData[T]["boloData_error"][cam]
    saved = list(err)
    err[2] = np.inf                                                  # leave one channel out of the fit
    data = np.asarray(fitted.fitData[T]["boloData"][cam], dtype=float)
    fitted.bestFits[T]["synthData"] = {"elongatedRing": {cam: 0.95 * data}}   # a model 5% low
    f, (ax, res) = plt.subplots(2, 1)
    try:
        fitted._plot_signal_panel(ax, cam, T, "[W/m2]", legend=True, resid_ax=res)
        assert "χ²/N" in ax.get_title() and "(7 channels)" in ax.get_title()
        hollow = [ln for ln in ax.get_lines() if ln.get_label() == "not fitted"]
        assert len(hollow) == 1 and len(hollow[0].get_xdata()) == 1
        points = [ln for ln in res.get_lines() if ln.get_marker() == "o"][0]
        assert len(points.get_xdata()) == 7
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "model" in legend and not any(t.endswith(" emission") for t in legend)
    finally:
        err[:] = saved
        plt.close(f)


def test_chord_numbers_can_be_thinned(fitted):
    tok = Tokamak(tokamakName="MOCK", mode="Build", loadBolometers=True, workspace=fitted.ws)
    f, ax = plt.subplots()
    try:
        tok._plot_bolometers(ax, "POL_A", label_every=5)
        assert sorted(t.get_text() for t in ax.texts) == ["1", "5"]
    finally:
        plt.close(f)


@pytest.mark.cherab
def test_best_fit_figure_is_saved_and_closed(tmp_path):
    from emis3d.mock import selftest

    selftest(workdir=tmp_path, pixel_samples=20, verbose=False)
    e = Emis3D(runConfigName=str(tmp_path / "runs" / "selftest" / "run.yaml"))
    e._perform_fits(evalTime=T)
    e._plot_bestFit(evalTime=T, save=True)
    assert list((tmp_path / "runs" / "selftest" / "results").glob("*/images/1_10.0000.png"))
    assert plt.get_fignums() == []
