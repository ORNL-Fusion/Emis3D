"""
tests/test_injector.py

SPI injectors from the machine settings (emis3d.injector), in the DREAM
runs' convention: the shatter point, the spray's centre line (angle_deg, or
aim_R/aim_Z, or the magnetic axis) and the full cone width divergence_deg,
plus the toroidal angle phi_deg. A run names one with `injector:`.
"""

import numpy as np
import pytest
import yaml

from emis3d.injector import Injector, load_injectors, machine_injectors
from emis3d.run_config import RunConfig
from emis3d.workspace import Workspace

BASE = {"phi_deg": 225.0, "shatter_R": 2.25, "shatter_Z": 0.35, "divergence_deg": 20.0}


def test_conventions_match_dream():
    inj = Injector.from_settings("A", {**BASE, "angle_deg": 0.0})
    assert inj.direction(1.7, 0.0) == pytest.approx(np.pi)                                # 0: horizontally inward
    up = Injector.from_settings("A", {**BASE, "angle_deg": 10.0})
    R, Z = up.ray(1.7, 0.0, length=1.0)
    assert R[-1] < R[0] and Z[-1] > Z[0]                                                    # +: inward and upward
    aimed = Injector.from_settings("A", BASE)                                               # neither: at the axis
    assert aimed.axis_miss(1.772, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert aimed.centre_angle_deg(1.772, 0.0) == pytest.approx(-np.degrees(np.arctan2(0.35, 2.25 - 1.772)))   # inward, downward
    at = Injector.from_settings("A", {**BASE, "aim_R": 1.5, "aim_Z": 0.35})
    assert at.direction(1.7, 0.0) == pytest.approx(np.pi) and at.centre_angle_deg(1.7, 0.0) == pytest.approx(0.0)


def test_cone_membership_and_axis_miss():
    inj = Injector.from_settings("A", {**BASE, "angle_deg": 0.0, "divergence_deg": 20.0})
    assert inj.in_cone(2.0, 0.35, 1.7, 0.0)                                                 # on the centre line
    assert inj.in_cone(2.0, 0.35 + 0.25 * np.tan(np.radians(9.5)), 1.7, 0.0)                # just inside the 10 deg half-angle
    assert not inj.in_cone(2.0, 0.35 + 0.25 * np.tan(np.radians(10.5)), 1.7, 0.0)
    assert not inj.in_cone(2.4, 0.35, 1.7, 0.0)                                             # behind the shatter point
    assert inj.axis_miss(1.7, 0.0) == pytest.approx(0.35)                                   # passes 35 cm above the axis


@pytest.mark.parametrize("entry, match", [
    ({**BASE, "divergence": 0.35}, "ambiguous"),
    ({**BASE, "angle": 0.0}, "unknown key"),
    ({"shatter_R": 2.0, "shatter_Z": 0.0}, "needs"),
    ({**BASE, "angle_deg": 0.0, "aim_R": 1.5}, "not both"),
    ({**BASE, "divergence_deg": 200.0}, "between 0 and 180"),
    ({**BASE, "shatter_R": "far"}, "must be a number"),
])
def test_bad_entries_are_refused(entry, match):
    with pytest.raises(ValueError, match=match):
        Injector.from_settings("X", entry)


def test_the_mock_machine_has_an_example_injector(tmp_path):
    ws = Workspace.resolve(env={}, user_config=tmp_path / "none.yaml")
    injectors = machine_injectors(ws, "MOCK")
    assert set(injectors) == {"SPI225"} and injectors["SPI225"].phi_deg == 225.0 and injectors["SPI225"].verified
    assert "cone 20 deg" in injectors["SPI225"].describe(1.772, 0.0)
    assert load_injectors({}) == {} and load_injectors({"INJECTORS": None}) == {}


def test_run_config_injector_sets_or_checks_the_injection_location(tmp_path):
    from emis3d.emis3D import Emis3D
    from test_calibration import write_workspace

    run = write_workspace(tmp_path, tstart=10.0, injector="SPI225")
    e = Emis3D(runConfigName=str(run))
    assert e.injector.name == "SPI225" and e.info["injectionLocation"] == 225.0

    cfg = yaml.safe_load(run.read_text())
    cfg["injectionLocation"] = 90.0
    run.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="does not match injector SPI225"):
        Emis3D(runConfigName=str(run))
    cfg["injectionLocation"], cfg["injector"] = 225.0, "NOPE"
    run.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="not in MOCK's settings"):
        Emis3D(runConfigName=str(run))
    with pytest.raises(ValueError, match="injector must be"):
        RunConfig.from_dict({"BOLOMETERS": {"A": {"dataFileName": "a.h5"}}, "shot": 1, "radDistDirectories_LocIndependent": ["x"], "injector": 3})


def test_check_reports_the_injector(tmp_path):
    from emis3d.pipeline import check
    from test_calibration import write_workspace

    run = write_workspace(tmp_path, tstart=10.0, injector="SPI225")
    msgs = [f.message for f in check(run)]
    assert any(m.startswith("SPI225: phi 225 deg") for m in msgs)


def test_spray_is_drawn_without_moving_the_view():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inj = Injector.from_settings("SPI225", BASE)
    fig, ax = plt.subplots()
    ax.set_xlim(1.0, 2.4)
    ax.set_ylim(-1.0, 1.0)
    inj.plot(ax, 1.772, 0.0)
    assert ax.get_xlim() == (1.0, 2.4) and ax.get_ylim() == (-1.0, 1.0)
    assert len(ax.lines) == 4 and ax.get_legend_handles_labels()[1] == ["SPI225 (20 deg cone)"]
    plt.close(fig)
