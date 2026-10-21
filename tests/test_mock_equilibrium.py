"""
tests/test_mock_equilibrium.py

The MOCK machine's Solov'ev equilibrium (emis3d.mock, inputs/equilibria):
it is an exact Grad-Shafranov solution inside the MOCK wall, its q profile
matches traced field lines, it loads without a workspace, and helical
radDists can be built on it.
"""

import tempfile

import numpy as np
import pytest
from matplotlib.path import Path as MPath

from emis3d import mock
from emis3d.fieldline_tracer import Fieldline_Tracer
from emis3d.tokamak import Tokamak
from emis3d.workspace import BUILTIN_MACHINES, Workspace, set_workspace


@pytest.fixture(scope="module")
def unconfigured():
    return Workspace.resolve(env={}, user_config=tempfile.mktemp())


@pytest.fixture(scope="module")
def g(unconfigured):
    return Tokamak(tokamakName="MOCK", mode="Analysis", eqFileName=mock.EQ_FILE, workspace=unconfigured).gfile


def grid(g):
    R = np.linspace(g.rleft, g.rleft + g.rdim, g.nx)
    Z = np.linspace(g.zmid - g.zdim / 2, g.zmid + g.zdim / 2, g.ny)
    return R, Z


def test_loads_from_the_machine_folder_without_a_workspace(unconfigured, g):
    assert unconfigured.layout == "unconfigured" and g is not None
    assert np.allclose(np.column_stack([g.rlim, g.zlim]), mock.wall_polygon(), atol=1e-8)   # same wall
    header = (BUILTIN_MACHINES / "MOCK" / "inputs" / "equilibria" / mock.EQ_FILE).read_text().splitlines()[0]
    assert "01/01/2026" in header                                    # fixed date: reproducible file


def test_is_a_grad_shafranov_solution(g):
    """Delta* psi = -mu0 R^2 p' with F F' = 0, at every interior grid point."""
    R, Z = grid(g)
    psi = np.asarray(g.psi)
    RR = R[:, None]
    dpsi_dR = np.gradient(psi, R[1] - R[0], axis=0)
    dstar = RR * np.gradient(dpsi_dR / RR, R[1] - R[0], axis=0) + np.gradient(np.gradient(psi, Z[1] - Z[0], axis=1), Z[1] - Z[0], axis=1)
    ratio = (dstar / RR**2)[3:-3, 3:-3]
    assert ratio == pytest.approx(-mock.MU0 * g.pprime[0], rel=1e-4)
    assert np.allclose(g.ffprime, 0.0) and np.allclose(g.fpol, g.fpol[0])


def test_plasma_sits_inside_the_wall(g):
    wall = MPath(np.column_stack([g.rlim, g.zlim]))
    assert wall.contains_points(np.column_stack([g.rbdry, g.zbdry])).all()
    assert wall.contains_point((g.rmagx, g.zmagx))
    R, Z = grid(g)
    psin = (np.asarray(g.psi) - g.simagx) / (g.sibdry - g.simagx)
    i, j = np.unravel_index(np.argmin(psin), psin.shape)
    assert R[i] == pytest.approx(g.rmagx, abs=R[1] - R[0]) and abs(Z[j]) <= Z[1] - Z[0]   # the axis is the psi minimum


@pytest.mark.parametrize("psin", [0.1, 0.5, 0.9])
def test_q_profile_matches_traced_field_lines(g, psin):
    sol = mock.solovev()
    Rs, _ = mock.flux_surface(sol, psin * sol["psi_b"])
    tr = Fieldline_Tracer(StartR=Rs[0], StartZ=0.0, StartPhi=0.0, gfile=g, NumTor=8000)
    tr.trace(NumTransits=16.0)
    th = np.unwrap(np.arctan2(tr.data["z"], tr.data["R"] - g.rmagx))
    th = np.abs(th - th[0])
    k = np.floor(th[-1] / (2 * np.pi))
    q_traced = np.interp(2 * np.pi * k, th, np.abs(tr.data["phi"])) / (2 * np.pi * k)
    assert q_traced == pytest.approx(np.interp(psin, np.linspace(0, 1, g.nx), g.qpsi), rel=1e-3)
    assert g.qpsi[0] == pytest.approx(mock.DEFAULT_EQ.q0)


def test_workspace_equilibrium_takes_priority(tmp_path):
    folder = tmp_path / "equilibria" / "MOCK"
    folder.mkdir(parents=True)
    text = (BUILTIN_MACHINES / "MOCK" / "inputs" / "equilibria" / mock.EQ_FILE).read_text()
    (folder / mock.EQ_FILE).write_text(text.replace("MOCK      ", "WORKSPACE ", 1))
    ws = Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml")
    tok = Tokamak(tokamakName="MOCK", mode="Analysis", eqFileName=mock.EQ_FILE, workspace=ws)
    assert tok.gfile is not None
    assert (folder / mock.EQ_FILE).read_text().startswith("WORKSPACE")   # the copy that was found first


def test_helical_raddist_starts_on_both_directions(unconfigured):
    """Both directions start at the start point, so their emissivity there is the same.
    (A revolution-splitting bug gave clockwise lines their end point instead.)"""
    from emis3d.radDist import Helical

    set_workspace(unconfigured)
    try:
        cfg = {"distType": "Helical", "tokamakName": "MOCK", "units": "Brightness", "injectionLocation": 225.0,
               "eqFileName": mock.EQ_FILE, "sigma_R": 0.1, "sigmaKernel": 0.05, "numFieldLines": 20,
               "numTransists": 1.0, "saveRunsDirectoryName": "h", "BOLOMETER_PROPS": {"pixelSamples": 10, "seed": 0}}
        h = Helical(startR=1.95, startZ=0.1, config=cfg)
        assert h.info["emissionNames"] == ["counterClock_rev0", "clockwise_rev0"]
        for deg in (224.9, 225.1):
            e = [h.calc_emissivity(np.array([1.95]), np.array([0.1]), phi=np.array([np.deg2rad(deg)]), emissionName=n)[n][0]
                 for n in h.info["emissionNames"]]
            assert e[0] > 1.0 and e[1] == pytest.approx(e[0], rel=0.05)
    finally:
        set_workspace(None)


def test_clockwise_revolutions_are_in_travel_order(unconfigured):
    tok = Tokamak(tokamakName="MOCK", mode="Analysis", eqFileName=mock.EQ_FILE, workspace=unconfigured)
    tok.set_fieldlines(startR=[1.95], startZ=[0.0], startPhi=np.deg2rad(225.0), numTransists=3.0)
    fl = tok.fieldLines[next(iter(tok.fieldLines))]
    for direction in ("counterClock", "clockwise"):
        starts = [np.asarray(fl[f"{direction}_rev{k}"]["phi"]).ravel()[0] for k in range(3)]
        assert np.allclose(np.rad2deg(starts), 225.0, atol=1e-6)      # each revolution starts at the injector
        first_R = np.asarray(fl[f"{direction}_rev0"]["R"]).ravel()[0]
        assert first_R == pytest.approx(1.95, abs=1e-6)               # rev0 begins at the start point
