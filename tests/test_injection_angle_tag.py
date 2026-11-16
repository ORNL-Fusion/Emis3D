"""
tests/test_injection_angle_tag.py

The injection location in LMFIT parameter names (Util_emis3D.loc_tag). A
negative angle, e.g. -135 deg for 225, used to give the name 'a_-135', which
LMFIT refuses ("not a valid Parameters name"), so the run stopped while
preparing the fit. The tag is now the angle in [0, 360).
"""

import json

import numpy as np
import pytest
from lmfit import Parameters

from emis3d import util_emis3D as Util_emis3D
from emis3d.emis3D import Emis3D
from test_calibration import write_workspace


@pytest.mark.parametrize("angle, tag", [(225, "225"), (225.0, "225"), (-135.0, "225"), (-0.2, "0"), (359.6, "0"), (360.0, "0"),
                                        (0.0, "0"), (11.4, "11"), (585.0, "225"), (-360.0, "0")])
def test_tag_is_the_angle_in_whole_degrees_from_0_to_359(angle, tag):
    assert Util_emis3D.loc_tag(angle) == tag
    Parameters().add(f"a_{Util_emis3D.loc_tag(angle)}", value=1.0)             # a name LMFIT accepts


def test_a_negative_injection_location_fits(tmp_path):
    run = write_workspace(tmp_path, tstart=10.0, scale_def="exponential", enable_dphi_scaling=True, injectionLocation=-135.0)
    for f in (tmp_path / "raddists" / "MOCK" / "ring").glob("*.json"):       # radDists made for an injector at -135 deg
        saved = json.loads(f.read_text())
        saved["info"]["injectionLocation"] = -135.0
        f.write_text(json.dumps(saved))
    e = Emis3D(runConfigName=str(run))
    e._prepare_fits(evalTime=10.0)
    e._minimize_radDists(evalTime=10.0)
    assert np.all(np.isfinite(e.fits[10.0]["scores"]))
    best = e.fits[10.0][int(np.argmin(e.fits[10.0]["scores"]))]
    assert {"a_225", "b_elongatedRing_225"} <= set(best["fit"].params)
    # the same parameters and the same fit as for the same location written as 225 deg
    synth = dict(best["synthetic_dict"], injectionLocation=225.0, injectionLocation_rad=np.deg2rad(225.0))
    again = Util_emis3D.fit_radDist(best["parameters"], e.fitData[10.0], synth, "exponential", 0.0)
    assert again.chisqr == pytest.approx(best["fit"].chisqr)
