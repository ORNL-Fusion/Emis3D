"""
tests/test_observed_phi_missing.py

A radDist file with signals but no observation angles (observed_phi_loc), as
NIMROD and M3DC1 radDists used to be saved. The fit silently placed every
such channel at phi = 0, so every camera got the same toroidal factor
whatever its angle. Now a scale that depends on the angle refuses the
radDist with a message saying how to fix it; scale_def: constant, which
does not use the angles, still works.
"""

import json

import pytest

from emis3d.globals import RADDIST_FORMAT
from emis3d.radDistFitting import RadDistFitting

CHANNELS = [["A1", "A2"], ["B1"]]


def raddist(path, angles: bool):
    data = {
        "Brightness": {"channelOrder": {"A": ["A1", "A2"], "B": ["B1"]}, "NIMROD": {"A": [[1.0], [2.0]], "B": [[3.0]]}},
        "Brightness_error": {"NIMROD": {"A": [[0.0], [0.0]], "B": [[0.0]]}},
        "observed_phi_loc": {"NIMROD": {"A": [[0.5], [0.5]], "B": [[2.5]]} if angles else {"A": [], "B": []}},
    }
    info = {"radDistFormat": RADDIST_FORMAT, "units": "Brightness", "emissionNames": ["NIMROD"], "injectionLocation": 225.0, "distType": "NIMROD"}
    path.write_text(json.dumps({"info": info, "data": data}))
    rd = RadDistFitting(radDistPath=str(path))
    assert rd._load_ok
    rd.prepare_for_fits(CHANNELS)
    return rd


@pytest.mark.parametrize("scale_def", ["exponential", "linear", "step"])
def test_a_scale_that_uses_the_angle_refuses_a_raddist_without_angles(tmp_path, scale_def):
    rd = raddist(tmp_path / "t_40.json", angles=False)
    with pytest.raises(ValueError) as err:
        rd.create_parameters(enable_dphi_scaling=True, scale_def=scale_def)
    msg = str(err.value)
    assert "t_40.json" in msg and "A1, A2, B1" in msg and "Regenerate" in msg and "scale_def: constant" in msg


def test_constant_scale_does_not_need_the_angles(tmp_path):
    rd = raddist(tmp_path / "t_40.json", angles=False)
    rd.create_parameters(scale_def="constant")
    assert list(rd.fitSynthetic["params"]["params"]) == ["a_225"]


def test_a_raddist_with_angles_is_unaffected(tmp_path):
    rd = raddist(tmp_path / "t_40.json", angles=True)
    assert rd._channels_without_phi == []
    rd.create_parameters(enable_dphi_scaling=True, scale_def="exponential")
    assert rd.fitSynthetic["NIMROD"]["observed_phi_loc"] == [[[0.5], [0.5]], [[2.5]]]


def test_a_channel_missing_from_the_raddist_is_not_reported_as_lacking_an_angle(tmp_path):
    """Channels the radDist does not have at all keep their own report (ERROR CHANNELS) and a zero signal."""
    rd = raddist(tmp_path / "t_40.json", angles=True)
    rd.prepare_for_fits([["A1", "A2", "A3"], ["B1"]])
    assert rd._channels_without_phi == [] and "A3" in rd.info["ERROR CHANNELS"]
    rd.create_parameters(enable_dphi_scaling=True, scale_def="exponential")
