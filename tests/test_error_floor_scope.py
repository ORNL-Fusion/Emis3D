"""
tests/test_error_floor_scope.py

The error floor is a fraction of a peak: each camera's own by default, so a
camera with a much lower signal level (another diagnostic, or far from the
radiation) still constrains the fit; error_floor_scope: global uses the
peak over every camera, which can make a dim camera's errors as large as
its signals. Per-camera error_floor and error_relative override the run's.
"""

import numpy as np
import pytest
import yaml

from emis3d.emis3D import Emis3D
from emis3d.run_config import RunConfig
from test_calibration import CAMS, write_workspace


def prepared(tmp_path, **extra):
    run = write_workspace(tmp_path, tstart=10.0, error_definition="floor_relative", **extra)
    e = Emis3D(runConfigName=str(run))
    e.data["observed"]["POL_B"]["DATA_CALIBRATED"] *= 0.1                     # one camera ten times dimmer
    e._prepare_data_for_fit(evalTime=10.0)
    order = e.channel_order["bolometer_order"]
    err = {cam: np.array(e.fitData[10.0]["observed_error"][order.index(cam)]) for cam in order}
    sig = {cam: np.array(e.fitData[10.0]["observed"][order.index(cam)]) for cam in order}
    return err, sig


def expected(sig, peak, floor=0.05, relative=0.05):
    return np.sqrt((floor * peak) ** 2 + (relative * sig) ** 2)


def test_floor_per_camera_keeps_a_dim_camera_in_the_fit(tmp_path):
    err, sig = prepared(tmp_path)
    for cam in ("POL_A", "POL_B"):
        np.testing.assert_allclose(err[cam], expected(sig[cam], sig[cam].max()), rtol=1e-9)   # each camera's own peak
    assert err["POL_B"].min() < 0.2 * err["POL_A"].min()                            # not swamped by the bright camera


def test_global_scope_reproduces_the_old_behaviour(tmp_path):
    err, sig = prepared(tmp_path, error_floor_scope="global")
    peak = max(v.max() for v in sig.values())                                      # over every camera
    np.testing.assert_allclose(err["POL_B"], expected(sig["POL_B"], peak), rtol=1e-9)
    assert err["POL_B"].min() > 0.4 * sig["POL_B"].max()                           # a dim camera's errors ~ its signals


def test_per_camera_error_overrides(tmp_path):
    err, sig = prepared(tmp_path, BOLOMETERS={c: {"dataFileName": f"1/{c}.h5", **({"error_floor": 0.2, "error_relative": 0.0} if c == "POL_B" else {})}
                                             for c in CAMS})
    np.testing.assert_allclose(err["POL_B"], expected(sig["POL_B"], sig["POL_B"].max(), 0.2, 0.0), rtol=1e-9)
    assert np.allclose(err["POL_B"], err["POL_B"].min())                          # no relative term for POL_B
    assert err["POL_A"].max() > err["POL_A"].min()                                 # POL_A keeps the run's relative term


def test_config_validation():
    base = {"BOLOMETERS": {"A": {"dataFileName": "a.h5"}}, "shot": 1, "radDistDirectories_LocIndependent": ["x"]}
    assert RunConfig.from_dict(base).error_floor_scope == "camera"
    with pytest.raises(ValueError, match="error_floor_scope"):
        RunConfig.from_dict({**base, "error_floor_scope": "everywhere"})
    with pytest.raises(ValueError, match="BOLOMETERS.A.error_floor"):
        RunConfig.from_dict({**base, "BOLOMETERS": {"A": {"dataFileName": "a.h5", "error_floor": -1}}})
    with pytest.raises(ValueError, match="unknown key"):
        RunConfig.from_dict({**base, "BOLOMETERS": {"A": {"dataFileName": "a.h5", "errorFloor": 0.1}}})
    assert yaml.safe_dump({"error_floor_scope": "camera"})
