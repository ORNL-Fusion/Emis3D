"""
tests/test_config_settings.py

Config values that used to be accepted but ignored now take effect:
- run config peak_shift_max_degrees limits the fitted peak shift
- radDist config integrationStep sets the ray-integration step
- radDist config numBins sets the toroidal points of the radiated-power profile
"""

import numpy as np
import pytest

from emis3d.emis3D import Emis3D
from emis3d.radDist import ElongatedRing
from emis3d.workspace import Workspace, set_workspace
from test_calibration import CAMS, write_workspace

RING = {"distType": "ElongatedRing", "tokamakName": "MOCK", "units": "Brightness", "injectionLocation": 11.5,
        "eqFileName": None, "sigma_R": 0.2, "sigma_z": 0.25, "rotationAngle": 0.0,
        "saveRunsDirectoryName": "x", "BOLOMETER_PROPS": {"pixelSamples": 10, "seed": 0}}


def test_peak_shift_limit_comes_from_the_run_config(tmp_path):
    run = write_workspace(tmp_path, miscal={c: 1.0 for c in CAMS}, scale_def="exponential",
                          vary_peak_rad_location=True, peak_shift_max_degrees=30.0)
    e = Emis3D(runConfigName=str(run))
    e._prepare_fits(evalTime=10.0)
    p = e.fits[10.0][0]["parameters"]
    assert p["peak_shift"].max == pytest.approx(np.deg2rad(30.0))
    assert p["peak_shift"].min == pytest.approx(-np.deg2rad(30.0))


def test_fractional_injection_angle_is_not_truncated(tmp_path):
    from emis3d.radDistFitting import RadDistFitting

    rd = RadDistFitting()
    rd.info["injectionLocation"] = 11.5
    rd.fitSynthetic = {"emissionNames": []}
    rd.info["emissionNames"] = []
    rd.create_parameters(vary_peak_rad_location=True, scale_def="exponential")
    assert rd.fitSynthetic["params"]["params"]["injector_phi"].value == pytest.approx(np.deg2rad(11.5))


@pytest.fixture
def mock_ws(tmp_path):
    set_workspace(Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml"))
    yield tmp_path
    set_workspace(None)


def test_integration_step_from_the_radDist_config(mock_ws):
    assert ElongatedRing(startR=1.75, startZ=0.15, config={**RING, "integrationStep": 0.02}).info["integrationStep"] == 0.02
    assert ElongatedRing(startR=1.75, startZ=0.15, config=dict(RING)).info["integrationStep"] == 0.01


def test_num_bins_sets_the_toroidal_profile_resolution(mock_ws):
    rd = ElongatedRing(startR=1.75, startZ=0.15, config={**RING, "numBins": 36})
    rd.calc_radiated_power()
    prof = rd.data["toroidalRadiatedPower"]["elongatedRing"]
    assert len(prof["phi_array"]) == len(prof["P_pol"]) == 36
