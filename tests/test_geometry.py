"""
tests/test_geometry.py

The geometry matrix G (emis3d.geometry): a linear forward model built once
per machine and cameras. Checked against Emis3D's ray tracing on MOCK, for
rings, helical radDists and a NIMROD radDist, skewed cameras included.
"""

import contextlib
import io
import json

import numpy as np
import pytest
import scipy.sparse as sp

import emis3d.radDist as RD
from emis3d import mock
from emis3d.cli import main
from emis3d.geometry import GeometryMatrix, GridSpec, build_geometry_matrix, matrix_path
from emis3d.workspace import Workspace, set_workspace

BASE = {"tokamakName": "MOCK", "units": "Brightness", "injectionLocation": 225.0, "saveRunsDirectoryName": "p",
        "BOLOMETER_PROPS": {"pixelSamples": 1500, "seed": 0}, "integrationStep": 0.01}
COARSE = {"dR": 0.05, "dz": 0.05, "n_phi": 72}
WALL = {"minr": 1.1, "maxr": 2.3, "minz": -1.0, "maxz": 1.0}


# ---------------------------------------------------------------------------
# Grid and matrix mechanics (no ray tracing)
# ---------------------------------------------------------------------------

def test_grid_presets_cover_the_wall():
    g = GridSpec.from_wall(WALL, "low")
    assert g.r_edges[0] == pytest.approx(1.08) and g.r_edges[-1] == pytest.approx(2.32)
    assert g.n_r == 25 and g.n_z == 41 and g.n_phi == 72 and g.n_voxels == 25 * 72 * 41
    assert "5.0 cm x 5.00 deg x 5.0 cm" in g.describe()
    assert GridSpec.from_wall(WALL, "high").n_voxels > GridSpec.from_wall(WALL, "medium").n_voxels > g.n_voxels
    assert GridSpec.from_wall(WALL, {"dR": 0.1, "dz": 0.2, "n_phi": 8}).n_phi == 8


def synthetic_matrix(tmp_path):
    g = GridSpec.from_wall(WALL, {"dR": 0.4, "dz": 0.5, "n_phi": 4})
    rng = np.random.default_rng(0)
    rows = sp.random(3, g.n_voxels, density=0.3, random_state=rng, format="csr")
    return GeometryMatrix(machine="MOCK", grid=g, rows=rows, cameras=["A", "B"], channels=["A1", "A2", "B1"],
                          camera_of_row=np.array([0, 0, 1]), etendues=np.array([2e-7, 3e-7, 4e-7]),
                          inside_wall=np.ones(g.n_voxels, bool), samples=10, provenance={"note": "synthetic"})


def test_units_follow_bolos_observe(tmp_path):
    G = synthetic_matrix(tmp_path)
    eps = np.ones(G.grid.n_voxels)
    b = G.signals(eps, "Brightness")
    np.testing.assert_allclose(G.signals(eps, "Radiance"), b / (4 * np.pi))
    np.testing.assert_allclose(G.signals(eps, "Power"), b * G.etendues / (4 * np.pi))
    with pytest.raises(ValueError, match="units"):
        G.signals(eps, "Watts")


def test_save_and_load_round_trip(tmp_path):
    G = synthetic_matrix(tmp_path)
    path = G.save(tmp_path / "g.npz")
    H = GeometryMatrix.load(path)
    assert (H.rows != G.rows).nnz == 0 and H.channels == G.channels and H.cameras == G.cameras
    np.testing.assert_array_equal(H.grid.r_edges, G.grid.r_edges)
    assert H.provenance == {"note": "synthetic"} and H.samples == 10


def test_voxel_index_order():
    G = synthetic_matrix(None)
    g = G.grid
    assert G.eps_index(0, 0, 0) == 0 and G.eps_index(0, 0, 1) == 1
    assert G.eps_index(0, 1, 0) == g.n_z and G.eps_index(1, 0, 0) == g.n_phi * g.n_z


# ---------------------------------------------------------------------------
# Against ray tracing on MOCK
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_G(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ws")
    set_workspace(Workspace.resolve(workspace=tmp, env={}, user_config=tmp / "none.yaml"))
    with contextlib.redirect_stdout(io.StringIO()):
        ring = RD.ElongatedRing(startR=1.85, startZ=0.1, config={**BASE, "distType": "ElongatedRing", "eqFileName": None,
                                                                 "sigma_R": 0.15, "sigma_z": 0.2, "rotationAngle": 30.0})
    G = build_geometry_matrix(ring.tokamak, resolution=COARSE, samples=4000)
    yield tmp, ring, G
    set_workspace(None)


def ratio(G_signals, radDist, name, cameras=None):
    rt = radDist.data["Brightness"][name]
    cams = [c for c in rt if cameras is None or c in cameras]
    g = np.concatenate([G_signals[name][c] for c in cams])
    r = np.concatenate([[float(np.sum(v)) for v in rt[c]] for c in cams])
    lit = r > 0.05 * r.max()
    q = g[lit] / r[lit]
    return q.mean(), q.std(ddof=1) / np.sqrt(q.size), int(lit.sum())


@pytest.mark.cherab
def test_matrix_matches_ray_tracing_for_a_ring(mock_G):
    _, ring, G = mock_G
    assert G.rows.shape == (40, G.grid.n_voxels) and len(G.channels) == 40 and G.cameras == ["POL_A", "POL_B", "POL_C", "SKEW", "WRAP"]
    sig = G.observe(ring)
    with contextlib.redirect_stdout(io.StringIO()):
        ring.bolos_observe()                                         # the foils' settings were restored after the build
    for cams in (["POL_A", "POL_B", "POL_C"], ["SKEW", "WRAP"]):
        m, err, n = ratio(sig, ring, "elongatedRing", cams)
        assert abs(m - 1.0) < 3 * err + 0.03, (cams, m, err, n)       # within noise and coarse-grid bias


@pytest.mark.cherab
def test_matrix_matches_ray_tracing_for_toroidally_varying_emission(mock_G):
    tmp, _, G = mock_G
    r, z, ph = np.linspace(1.05, 2.35, 27), np.linspace(-1.05, 1.05, 43), np.linspace(0, 2 * np.pi, 36, endpoint=False)
    Z, R, P = np.meshgrid(z, r, ph, indexing="ij")
    prad = np.exp(-((R - 1.85) ** 2 / 0.08 + (Z - 0.1) ** 2 / 0.125)) * (1 + 0.8 * np.cos(P - np.deg2rad(225)))
    (tmp / "sim.json").write_text(json.dumps({"pradArray": prad.tolist(), "r_list": r.tolist(), "z_list": z.tolist(), "phi_list": ph.tolist()}))
    with contextlib.redirect_stdout(io.StringIO()):
        nim = RD.NIMROD(nimrodFile=str(tmp / "sim.json"), timestep=1, config={**BASE, "eqFileName": None})
        sig = G.observe(nim)
        nim.bolos_observe()
    m, err, n = ratio(sig, nim, "NIMROD")                            # all five cameras, skewed ones included
    assert abs(m - 1.0) < 3 * err + 0.03 and n > 20

    with contextlib.redirect_stdout(io.StringIO()):
        hel = RD.Helical(startR=1.95, startZ=0.1, config={**BASE, "distType": "Helical", "eqFileName": mock.EQ_FILE,
                                                          "sigma_R": 0.1, "sigmaKernel": 0.05, "numFieldLines": 30, "numTransists": 1.0})
        sig = G.observe(hel)
        hel.tokamak.bolometers = [b for b in hel.tokamak.bolometers if b.name.startswith("POL_")]
        hel.bolos_observe()
    for name in hel.info["emissionNames"]:
        m, err, n = ratio(sig, hel, name)
        assert abs(m - 1.0) < 3 * err + 0.05, (name, m, err, n)


@pytest.mark.cherab
def test_build_matrix_command_saves_to_the_cache(tmp_path, capsys):
    (tmp_path / "emis3d.yaml").write_text("version: 1\n")
    assert main(["build-matrix", "MOCK", "--resolution", "low", "--samples", "500", "--cameras", "POL_A,SKEW", "--workspace", str(tmp_path)]) == 0
    ws = Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml")
    path = matrix_path(ws, "MOCK", "low_POL_A_SKEW")
    assert path.is_file() and "saved" in capsys.readouterr().out
    G = GeometryMatrix.load(path)
    assert G.cameras == ["POL_A", "SKEW"] and G.rows.shape[0] == 16 and G.samples == 500
    assert "commit" in G.provenance or "emis3d" in str(G.provenance)
