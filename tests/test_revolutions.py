"""
tests/test_revolutions.py

Review issue 14: helical revolutions are counted once.

A field line traced for 10 transits through a circular test equilibrium on
MOCK, in both directions. For every revolution k, the toroidal distance that
the fit uses to reach a camera (find_dphi with the emission name
'<direction>_rev<k>') must equal the distance actually travelled along the
line, read from a continuous trace. Before the fix the fit added an extra
2*pi*k.
"""

import numpy as np
import pytest
import yaml
from freeqdsk import geqdsk

import emis3d.util_emis3D as U
from emis3d.fieldline_tracer import Fieldline_Tracer
from emis3d.radDist import Helical
from emis3d.tokamak import Tokamak
from emis3d.workspace import BUILTIN_MACHINES, Workspace
from test_fieldline_tracer import R0, make_gfile

MU = np.deg2rad(225.0)
START_R = R0 + 0.3
N = 10
GFILE_KEYS = ("nx", "ny", "rdim", "zdim", "rcentr", "rleft", "zmid", "rmagx", "zmagx", "simagx", "sibdry",
              "bcentr", "cpasma", "fpol", "pres", "ffprime", "pprime", "psi", "qpsi", "nbdry", "nlim",
              "rbdry", "zbdry", "rlim", "zlim")


@pytest.fixture(scope="module")
def tokamak(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    (root / "equilibria" / "MOCK").mkdir(parents=True)
    g = make_gfile(eps=0.05)
    with open(root / "equilibria" / "MOCK" / "circ.geqdsk", "w") as fh:
        geqdsk.write({k: getattr(g, k) for k in GFILE_KEYS}, fh)
    ws = Workspace.resolve(workspace=root, env={}, user_config=root / "none.yaml")
    return Tokamak(tokamakName="MOCK", mode="Analysis", eqFileName="circ.geqdsk", workspace=ws)


@pytest.fixture(scope="module")
def traced(tokamak):
    tokamak.set_fieldlines(startR=[START_R], startZ=[0.0], startPhi=MU, numTransists=float(N))
    return tokamak.fieldLines[next(iter(tokamak.fieldLines))]


def camera_phi(name="POL_A"):
    cfg = yaml.safe_load((BUILTIN_MACHINES / "MOCK" / "inputs" / "sxrInfo" / f"{name}_config.yaml").read_text())
    return np.deg2rad(cfg["CAMERA_POSITION_R_Z_PHI"][2])


def test_every_revolution_is_stored_at_full_resolution(traced):
    names = traced["directionNames"]
    for direction in ("counterClock", "clockwise"):
        assert [n for n in names if n.startswith(direction)] == [f"{direction}_rev{k}" for k in range(N)]
    for name in names:
        assert len(traced[name]["phi"]) >= 499          # ~500 points per transit, whatever N is


@pytest.mark.parametrize("direction, sign", [("counterClock", +1.0), ("clockwise", -1.0)])
def test_distance_to_a_camera_is_counted_once(tokamak, direction, sign):
    tr = Fieldline_Tracer(StartR=START_R, StartZ=0.0, StartPhi=MU, gfile=tokamak.gfile, NumTor=500 * N)
    tr.trace(NumTransits=sign * N)
    travelled = np.abs(tr.data["phi"] - MU)             # continuous, along the line
    phi_cam = camera_phi()
    # distances at which the line passes the camera, in order
    first = (sign * (phi_cam - MU)) % (2 * np.pi)
    passes = [first + 2 * np.pi * k for k in range(N) if first + 2 * np.pi * k <= travelled[-1]]
    assert len(passes) == N

    hel = Helical.__new__(Helical)
    hel.info = {}
    info = {"NUM_CHANNELS": 1, "CAMERA_POSITION_R_Z_PHI": [2.3, 0.8, np.rad2deg(phi_cam)]}
    for k, true in enumerate(passes):
        name = f"{direction}_rev{k}"
        stored = hel._observed_phi_loc(info, name)[0][0]
        assert 0.0 <= stored < 2 * np.pi                 # a physical angle, the same for every revolution
        assert U.find_dphi(np.array([stored]), MU, name)[0] == pytest.approx(true, abs=1e-9)


def test_linear_scale_never_rises_again_along_the_line():
    phi_cam = camera_phi()
    d = np.array([U.find_dphi(np.array([phi_cam]), MU, f"counterClock_rev{k}")[0] for k in range(N)])
    s = U.scale_linear(1.0, 0.02, d)
    assert np.all(np.diff(d) == pytest.approx(2 * np.pi))
    assert np.all(np.diff(s) <= 0) and s[-1] == 0.0


def test_fractional_transits_keep_the_last_partial_revolution(tokamak):
    tokamak.set_fieldlines(startR=[START_R], startZ=[0.0], startPhi=np.deg2rad(45.0), numTransists=2.5)
    fl = tokamak.fieldLines[next(k for k in tokamak.fieldLines if np.isclose(float(k), 45.0))]
    assert [n for n in fl["directionNames"] if n.startswith("counterClock")] == [
        "counterClock_rev0", "counterClock_rev1", "counterClock_rev2"]


def test_non_positive_transits_are_rejected(tokamak):
    with pytest.raises(ValueError, match="positive"):
        tokamak.set_fieldlines(startR=[START_R], startZ=[0.0], startPhi=MU, numTransists=0.0)
