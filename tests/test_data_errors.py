"""
tests/test_data_errors.py

use_data_errors: the per-channel errors saved with the measurements
(DATA_CALIBRATED_ERROR, written by scripts/process_bolometer_data.py and
the DIII-D saver) raise the floor of the floor_relative error model:

    sigma^2 = max(error_floor * peak, saved error)^2 + (error_relative * signal)^2

The saved errors used to be written and never read.
"""

import h5py
import numpy as np
import pytest

from emis3d.emis3D import Emis3D
from emis3d.run_config import RunConfig
from test_calibration import CAMS, TIMES, write_workspace

T = 10.0
BASE = {"BOLOMETERS": {"A": {"dataFileName": "a.h5"}}, "shot": 1, "radDistDirectories_LocIndependent": ["x"]}


def add_errors(root, value_of_channel, cams=CAMS):
    """Add DATA_CALIBRATED_ERROR to the measurement files: value_of_channel(camera, i) for every sample of channel i."""
    for cam in cams:
        with h5py.File(root / "measurements" / "MOCK" / "1" / f"{cam}.h5", "a") as f:
            n = f["DATA_CALIBRATED"].shape[0]
            f.create_dataset("DATA_CALIBRATED_ERROR", data=np.array([[value_of_channel(cam, i)] * TIMES.size for i in range(n)], dtype=float))


def errors(root, saved=None, **config):
    root.mkdir(exist_ok=True)
    run = write_workspace(root, tstart=T, miscal={c: 1.0 for c in CAMS}, error_definition="floor_relative", **config)
    if saved is not None:
        add_errors(root, saved)
    e = Emis3D(runConfigName=str(run))
    e._prepare_data_for_fit(evalTime=T)
    order = e.channel_order["bolometer_order"]
    return ({c: np.array(e.fitData[T]["observed_error"][order.index(c)]) for c in order},
            {c: np.array(e.fitData[T]["observed"][order.index(c)]) for c in order})


def test_saved_errors_raise_the_floor(tmp_path):
    # channel 0 of every camera has a large saved error (above the floor), the others a tiny one (below it)
    err, sig = errors(tmp_path, saved=lambda cam, i: 400.0 if i == 0 else 1.0e-3, use_data_errors=True)
    for cam in CAMS:
        floor = 0.05 * sig[cam].max()
        assert floor < 400.0
        expected = np.sqrt(np.maximum(floor, np.where(np.arange(sig[cam].size) == 0, 400.0, 1.0e-3)) ** 2 + (0.05 * sig[cam]) ** 2)
        np.testing.assert_allclose(err[cam], expected, rtol=1e-9)


def test_off_by_default_even_when_the_files_have_errors(tmp_path):
    with_file, sig = errors(tmp_path / "a", saved=lambda cam, i: 400.0)
    for cam in CAMS:
        np.testing.assert_allclose(with_file[cam], np.sqrt((0.05 * sig[cam].max()) ** 2 + (0.05 * sig[cam]) ** 2), rtol=1e-9)


def test_saved_errors_alone(tmp_path):
    """error_floor 0 and error_relative 0: the saved errors are the errors."""
    err, _ = errors(tmp_path, saved=lambda cam, i: 7.0 + i, use_data_errors=True, error_floor=0.0, error_relative=0.0)
    for cam in CAMS:
        np.testing.assert_allclose(err[cam], 7.0 + np.arange(err[cam].size), rtol=1e-9)


def test_saved_errors_are_scaled_with_the_data(tmp_path):
    bolos = {c: {"dataFileName": f"1/{c}.h5", "scalingFactor": 3.0 if c == "POL_B" else 1.0} for c in CAMS}
    err, _ = errors(tmp_path, saved=lambda cam, i: 7.0, use_data_errors=True, error_floor=0.0, error_relative=0.0, BOLOMETERS=bolos)
    assert np.allclose(err["POL_A"], 7.0) and np.allclose(err["POL_B"], 21.0)


def test_the_error_is_the_rms_over_the_averaging_window(tmp_path):
    run = write_workspace(tmp_path, tstart=T, miscal={c: 1.0 for c in CAMS}, error_definition="floor_relative", use_data_errors=True,
                          error_floor=0.0, error_relative=0.0)
    add_errors(tmp_path, lambda cam, i: 1.0)
    e = Emis3D(runConfigName=str(run))
    trace = e.data["observed"]["POL_A"]["DATA_CALIBRATED_ERROR"]
    trace[:] = (1.0 + np.arange(TIMES.size))[None, :]                           # a different error at every sample
    inside = np.flatnonzero((TIMES >= T - 0.05 - 1e-9) & (TIMES <= T + 0.05 + 1e-9))
    np.testing.assert_allclose(e._data_error("POL_A", T), np.sqrt(np.mean((1.0 + inside) ** 2.0)))


def test_a_channel_without_a_saved_error_keeps_the_runs_floor(tmp_path):
    err, sig = errors(tmp_path, saved=lambda cam, i: np.nan if i == 2 else 400.0, use_data_errors=True)
    floor = 0.05 * sig["POL_A"].max()
    assert err["POL_A"][2] == pytest.approx(np.sqrt(floor**2 + (0.05 * sig["POL_A"][2]) ** 2))
    assert np.all(np.isfinite(err["POL_A"]))


def test_combined_with_a_noise_window_the_larger_wins(tmp_path):
    with_noise, _ = errors(tmp_path / "a", noise_window=[9.0, 9.5], error_floor=1e-6, error_relative=0.0)
    both, _ = errors(tmp_path / "b", saved=lambda cam, i: 1.0e-3 if i else 400.0, noise_window=[9.0, 9.5], use_data_errors=True,
                     error_floor=1e-6, error_relative=0.0)
    assert both["POL_A"][0] == pytest.approx(400.0)                              # the saved error is the larger
    np.testing.assert_allclose(both["POL_A"][1:], with_noise["POL_A"][1:], rtol=1e-9)   # the measured noise is


def test_a_file_without_errors_is_named(tmp_path):
    run = write_workspace(tmp_path, tstart=T, error_definition="floor_relative", use_data_errors=True)
    add_errors(tmp_path, lambda cam, i: 1.0, cams=[c for c in CAMS if c != "POL_C"])
    with pytest.raises(ValueError, match=r"use_data_errors: .*POL_C\.h5 has no DATA_CALIBRATED_ERROR"):
        Emis3D(runConfigName=str(run))


def test_config_validation():
    assert RunConfig.from_dict(BASE).use_data_errors is False
    assert RunConfig.from_dict({**BASE, "use_data_errors": True, "error_floor": 0.0}).use_data_errors is True   # no floor needed with it
    with pytest.raises(ValueError, match="use_data_errors needs error_definition floor_relative"):
        RunConfig.from_dict({**BASE, "use_data_errors": True, "error_definition": "poisson"})
    with pytest.raises(ValueError, match="true or false"):
        RunConfig.from_dict({**BASE, "use_data_errors": "yes"})
    with pytest.raises(ValueError, match="noise_window or use_data_errors"):
        RunConfig.from_dict({**BASE, "error_floor": 0.0})
