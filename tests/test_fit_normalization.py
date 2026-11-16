"""
tests/test_fit_normalization.py

The synthetic signals are normalized to the peak of the data, and the
amplitude's bounds are relative to that peak. The peak is taken over the
channels in the fit only: a bad channel (listed under badChannels, or NaN
for the whole shot) must not set it.

(It used to be the largest value in the data, bad channels included: a
saturated channel listed under badChannels rescaled every radDist and
pushed the true amplitude outside its bounds, and a channel that was NaN
for the whole shot made every fit fail if it was the first one.)
"""

import numpy as np
import pytest

from emis3d.emis3D import Emis3D
from test_calibration import CAMS, write_workspace

T = 10.0


def fit(tmp_path, bad=None, spoil=None):
    """Fit at T. bad: channels listed under badChannels of POL_A; spoil(e) changes the loaded data first."""
    bolos = {c: {"dataFileName": f"1/{c}.h5"} for c in CAMS}
    if bad:
        bolos["POL_A"]["badChannels"] = list(bad)
    tmp_path.mkdir(exist_ok=True)
    run = write_workspace(tmp_path, tstart=T, miscal={c: 1.0 for c in CAMS}, BOLOMETERS=bolos)
    e = Emis3D(runConfigName=str(run))
    if spoil:
        spoil(e)
    e._prepare_fits(evalTime=T)
    e._minimize_radDists(evalTime=T)
    best = int(np.argmin(e.fits[T]["scores"]))
    return e, e.fits[T][best]


def test_peak_is_over_the_channels_in_the_fit(tmp_path):
    def saturate(e):
        e.data["observed"]["POL_A"]["DATA_CALIBRATED"][0, :] = 1.0e6

    e, _ = fit(tmp_path, bad=["POL_A01"], spoil=saturate)
    good = [v for g, errs in zip(e.fitData[T]["observed"], e.fitData[T]["observed_error"], strict=True)
            for v, err in zip(g, errs, strict=True) if np.isfinite(err)]
    assert e._fitted_data_peak(T) == pytest.approx(max(good))
    assert e._fitted_data_peak(T) < 1.0e4                                      # not the bad channel's 1e6


def test_a_listed_bad_channel_does_not_change_the_fit(tmp_path):
    def saturate(e):
        e.data["observed"]["POL_A"]["DATA_CALIBRATED"][0, :] = 1.0e6

    _, clean = fit(tmp_path / "clean", bad=["POL_A01"])
    _, spoiled = fit(tmp_path / "spoiled", bad=["POL_A01"], spoil=saturate)
    assert spoiled["info"]["radDists"]["radDistNumber"] == clean["info"]["radDists"]["radDistNumber"]
    assert spoiled["synthetic_dict"]["elongatedRing"]["scaleSynth"] == pytest.approx(clean["synthetic_dict"]["elongatedRing"]["scaleSynth"])
    assert spoiled["fit"].params["a_0"].value == pytest.approx(clean["fit"].params["a_0"].value)
    assert spoiled["fit"].chisqr == pytest.approx(clean["fit"].chisqr)


def test_a_first_channel_without_data_does_not_break_the_fit(tmp_path):
    def blank(e):                                                              # first channel of the first camera
        e.data["observed"]["POL_A"]["DATA_CALIBRATED"][0, :] = np.nan

    e, best = fit(tmp_path, spoil=blank)
    assert np.isfinite(e._fitted_data_peak(T))
    assert np.isfinite(best["fit"].chisqr) and np.all(np.isfinite(e.fits[T]["scores"]))
    assert best["fit"].ndata == sum(len(g) for g in e.channel_order["channel_list"]) - 1   # the blank channel is left out


def test_no_usable_channel_is_reported(tmp_path):
    def blank_all(e):
        for cam in e.data["observed"].values():
            cam["DATA_CALIBRATED"][:] = np.nan

    with pytest.raises(ValueError, match="No channel in the fit has a positive signal"):
        fit(tmp_path, spoil=blank_all)
