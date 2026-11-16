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
from emis3d.radDist import Helical, HelicalRing
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


# ---------------------------------------------------------------------------
# A fractional last revolution emits only where the field line goes
# ---------------------------------------------------------------------------

HALF = np.deg2rad(90.0)  # start angle of a line traced for 1.5 transits: its second revolution is half a turn


@pytest.fixture(scope="module")
def half_turn(tokamak):
    tokamak.set_fieldlines(startR=[START_R], startZ=[0.0], startPhi=HALF, numTransists=1.5)
    return tokamak


def covers(tok, name, degrees):
    return tok.fieldline_covers("90", name, np.deg2rad(degrees)).tolist()


def test_whole_revolutions_reach_every_angle(half_turn):
    every = np.arange(0.0, 360.0, 0.25)
    for name in ("counterClock_rev0", "clockwise_rev0"):
        assert all(covers(half_turn, name, every)), name


def test_partial_revolution_reaches_only_the_angles_it_travels(half_turn):
    # counterclockwise from 90 deg for half a turn: 90 -> 270
    assert covers(half_turn, "counterClock_rev1", [90.0, 100.0, 180.0, 269.0, 270.0]) == [True] * 5
    assert covers(half_turn, "counterClock_rev1", [272.0, 315.0, 0.0, 45.0, 88.0]) == [False] * 5
    # clockwise from 90 deg for half a turn: 90 -> 0 -> 270
    assert covers(half_turn, "clockwise_rev1", [90.0, 45.0, 0.0, 315.0, 271.0, 270.0]) == [True] * 6
    assert covers(half_turn, "clockwise_rev1", [92.0, 135.0, 180.0, 268.0]) == [False] * 4
    # angles given beyond one turn are the same angles
    assert covers(half_turn, "counterClock_rev1", [180.0 + 360.0, 0.0 - 360.0]) == [True, False]


def _helical(tok, cls):
    rd = cls.__new__(cls)
    rd.tokamak = tok
    rd.info = {"startPhi": 90.0, "weights": [1.0], "sigmaKernel": 0.05, "sigma_R": 0.5, "sigma_z": 0.1}
    return rd


@pytest.mark.parametrize("cls", [Helical, HelicalRing])
def test_no_emission_where_a_partial_revolution_does_not_reach(half_turn, cls):
    """
    find_RZ_Fline answers at every angle with the nearest sample, so the half
    revolution used to emit all the way round, as a ring at its end points.
    """
    rd = _helical(half_turn, cls)

    def on_the_line(name, deg):
        phi = np.array([np.deg2rad(deg)])
        R, z = half_turn.find_RZ_Fline("90", name, inputPhis=phi)      # where the lookup puts the line at this angle
        return float(rd._evaluate(np.array([R[0, 0] + 0.01]), np.array([z[0, 0]]), phi, emissionName=name)[name][0])

    for deg in (100.0, 180.0, 260.0):                                   # reached by the counterclockwise half turn
        assert on_the_line("counterClock_rev1", deg) > 0.0
        assert on_the_line("clockwise_rev1", deg) == 0.0
    for deg in (280.0, 0.0, 80.0):                                      # reached by the clockwise one
        assert on_the_line("counterClock_rev1", deg) == 0.0
        assert on_the_line("clockwise_rev1", deg) > 0.0
    for deg in (0.0, 100.0, 180.0, 280.0):                              # the whole first revolutions emit everywhere
        assert on_the_line("counterClock_rev0", deg) > 0.0 and on_the_line("clockwise_rev0", deg) > 0.0


def test_helical_emission_at_several_angles_at_once(half_turn):
    """Points at reached and unreached angles in one call are each treated at their own angle."""
    rd = _helical(half_turn, Helical)
    phis = np.deg2rad([180.0, 0.0, 100.0, 300.0])
    R, z = half_turn.find_RZ_Fline("90", "counterClock_rev1", inputPhis=phis)
    out = rd._evaluate(R[0], z[0], phis, emissionName="counterClock_rev1")["counterClock_rev1"]
    assert (out > 0).tolist() == [True, False, True, False]
