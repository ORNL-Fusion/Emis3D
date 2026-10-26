"""
tests/test_inversion_tool.py

The standalone inversion tool (tools/inversion/invert_emissivity.py): its
grid, Laplacian, kernels, field-line mapping and solver, on a small
synthetic geometry matrix (no ray tracing).
"""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from emis3d import mock
from emis3d.geometry import GeometryMatrix, GridSpec
from emis3d.tokamak import Tokamak
from emis3d.workspace import Workspace

TOOL = Path(__file__).parents[1] / "tools" / "inversion" / "invert_emissivity.py"
spec = importlib.util.spec_from_file_location("invert_emissivity", TOOL)
inv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inv)


@pytest.fixture(scope="module")
def mock_tok():
    ws = Workspace.resolve(env={}, user_config=Path(tempfile.mktemp()))
    return Tokamak(tokamakName="MOCK", mode="Analysis", eqFileName=mock.EQ_FILE, workspace=ws)


def small_G(tok, n_channels=30, seed=0, dR=0.1, n_phi=24):
    g = GridSpec.from_wall(tok.wall, {"dR": dR, "dz": dR, "n_phi": n_phi})
    RR, ZZ = np.meshgrid(g.r_centers, g.z_centers, indexing="ij")
    inside_rz = tok._inside_tokamak(np.column_stack([RR.ravel(), ZZ.ravel()])).reshape(g.n_r, g.n_z)
    inside = np.repeat(inside_rz[:, None, :], g.n_phi, axis=1).ravel()
    rows = sp.random(n_channels, g.n_voxels, density=0.05, random_state=np.random.default_rng(seed), format="csr")
    return GeometryMatrix(machine="MOCK", grid=g, rows=rows, cameras=["A"], channels=[f"A{i:02d}" for i in range(n_channels)],
                          camera_of_row=np.zeros(n_channels, int), etendues=np.ones(n_channels), inside_wall=inside, samples=1)


def test_pixels_and_laplacian(mock_tok):
    G = small_G(mock_tok)
    ir, iz, index = inv.pixel_grid(G)
    assert ir.size == (index >= 0).sum() and np.all(index[ir, iz] == np.arange(ir.size))
    L = inv.laplacian(ir, iz, index)
    np.testing.assert_allclose(L @ np.ones(ir.size), 0.0, atol=1e-12)                 # constants are not rough
    assert (L.diagonal() <= 4).all() and np.allclose((L - L.T).toarray(), 0.0)


def test_axisymmetric_kernel_sums_the_sectors(mock_tok):
    G = small_G(mock_tok)
    ir, iz, _ = inv.pixel_grid(G)
    K = inv.kernel_axisymmetric(G, np.arange(len(G.channels)), ir, iz)
    x = np.random.default_rng(1).uniform(size=ir.size)
    eps = np.zeros(G.grid.n_voxels)
    for p, (r, z) in enumerate(zip(ir, iz, strict=True)):
        eps[G.eps_index(r, np.arange(G.grid.n_phi), z)] = x[p]
    np.testing.assert_allclose(K @ x, G.signals(eps), rtol=1e-12)


def test_solver_recovers_a_smooth_source_and_respects_positivity(mock_tok):
    G = small_G(mock_tok, n_channels=60, seed=2)
    ir, iz, index = inv.pixel_grid(G)
    L = inv.laplacian(ir, iz, index)
    K = inv.kernel_axisymmetric(G, np.arange(len(G.channels)), ir, iz)
    Rp, Zp = G.grid.r_centers[ir], G.grid.z_centers[iz]
    x_true = np.exp(-((Rp - 1.8) ** 2 + (Zp - 0.1) ** 2) / (2 * 0.25**2))
    signal = np.asarray(K @ x_true).ravel()
    sig = 0.02 * signal.max() + 0.02 * signal
    chosen, results = inv.scan(K, signal, sig, L, list(np.logspace(-6, 4, 11)), 1.0)
    lam, x, c, _ = chosen
    assert (x >= 0).all() and c <= 1.0
    assert all(r[2] <= results[-1][2] + 1e-9 for r in results)                            # weaker lambda fits at least as well
    assert np.corrcoef(x, x_true)[0, 1] > 0.8


def test_field_aligned_mapping(mock_tok):
    from emis3d.fieldline_map import FieldLineMap

    G = small_G(mock_tok, dR=0.15, n_phi=12)
    phi0 = float(G.grid.phi_centers[3])                                                       # on a sector centre
    fmap = FieldLineMap.build(G, mock_tok.gfile, phi0, "test", mock.EQ_FILE, transits=1, substeps=1)
    M = fmap.matrix(0.0)
    assert M.shape == (G.grid.n_voxels, fmap.n_pixels) and M.nnz > 0 and (M.data > 0).all()
    assert np.asarray(M.sum(axis=1)).max() <= 2.0 + 1e-9                                       # one pass per direction, each averaged
    hits = 0
    for p, (r, z) in enumerate(zip(fmap.ir, fmap.iz, strict=True)):
        row = M[G.eps_index(r, 3, z)]
        if row.nnz and row[0, p] > 0:
            hits += 1
    assert hits >= 0.9 * fmap.n_pixels                                                        # on the source plane each pixel maps to itself
    assert fmap.matrix(1.0).sum() < M.sum()                                                    # decay weakens the far sectors
    src = fmap.gaussian_source(1.8, 0.1, 0.2, 0.2)
    np.testing.assert_allclose(fmap.apply(src, 0.5), fmap.matrix(0.5) @ src, rtol=1e-12)


def test_mapping_round_trip_and_cache(mock_tok, tmp_path):
    from emis3d.fieldline_map import FieldLineMap, cached_map, map_path

    G = small_G(mock_tok, dR=0.2, n_phi=8)
    fmap = FieldLineMap.build(G, mock_tok.gfile, 0.3, "t", mock.EQ_FILE, transits=1, substeps=1)
    path = fmap.save(tmp_path / "m.npz")
    back = FieldLineMap.load(path)
    assert back.n_pixels == fmap.n_pixels and len(back.passes) == len(fmap.passes) and back.phi0 == pytest.approx(0.3)
    src = np.random.default_rng(0).uniform(size=fmap.n_pixels)
    np.testing.assert_allclose(back.apply(src, 0.2), fmap.apply(src, 0.2))
    assert cached_map(path) is cached_map(path)
    ws = Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml")
    assert map_path(ws, "MOCK", "low", "g_file.geqdsk", 225.0, 1).name == "low_map_g_file_225deg_1t.npz"


def test_solver_reaches_the_constrained_optimum(mock_tok):
    """L-BFGS-B must drive the projected gradient to zero, at strong and at weak regularization."""
    G = small_G(mock_tok, n_channels=60, seed=2)
    ir, iz, index = inv.pixel_grid(G)
    L = inv.laplacian(ir, iz, index)
    K = inv.kernel_axisymmetric(G, np.arange(len(G.channels)), ir, iz)
    Rp, Zp = G.grid.r_centers[ir], G.grid.z_centers[iz]
    x_true = 1e6 * np.exp(-((Rp - 1.9) ** 2 + (Zp - 0.6) ** 2) / (2 * 0.1**2))        # bright and compact
    d = np.asarray(K @ x_true).ravel()
    sig = 0.05 * d.max() + 0.05 * d
    prob = inv.Problem(K, d, sig, L)
    x = None
    for lam in (1e2, 1e-2, 1e-6):
        x, its, opt = prob.solve(lam, x0=x)
        assert (x >= 0).all() and opt < 1e-4, (lam, opt)
    assert inv.chi2_per_point(K, x, d, sig) < 1e-3                                      # an unregularized fit is exact


def test_inconsistent_views_are_reported_per_camera(mock_tok):
    """Data from a toroidally localized source cannot be fitted axisymmetrically; the per-camera chi2 says which views disagree."""
    G = small_G(mock_tok, n_channels=40, seed=3)
    G.cameras, G.camera_of_row = ["A", "B"], np.repeat([0, 1], 20)
    ir, iz, index = inv.pixel_grid(G)
    L = inv.laplacian(ir, iz, index)
    K = inv.kernel_axisymmetric(G, np.arange(40), ir, iz)
    eps = np.zeros(G.grid.n_voxels)
    sector = G.sector_of_voxel()
    Rv = np.repeat(np.repeat(G.grid.r_centers, G.grid.n_phi), G.grid.n_z)
    Zv = np.tile(G.grid.z_centers, G.grid.n_r * G.grid.n_phi)
    eps = np.exp(-((Rv - 1.8) ** 2 + (Zv - 0.2) ** 2) / (2 * 0.15**2)) * (sector < G.grid.n_phi // 4)   # one toroidal quarter only
    d = G.signals(eps)
    sig = 0.05 * d.max() + 0.05 * d
    chosen, results = inv.scan(K, d, sig, L, [1e-8, 1e-2], 1.0)
    cams = ["A"] * 20 + ["B"] * 20
    by_cam = inv.per_camera_chi2(K, chosen[1], d, sig, cams)
    assert set(by_cam) == {"A", "B"} and all(v >= 0 for v in by_cam.values())
    assert chosen[2] > 0 and len(results) == 2
