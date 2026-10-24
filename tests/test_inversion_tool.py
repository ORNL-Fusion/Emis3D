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
    G = small_G(mock_tok, dR=0.15, n_phi=12)
    ir, iz, _ = inv.pixel_grid(G)
    phi0 = float(G.grid.phi_centers[3])                                                       # on a sector centre
    M = inv.mapping_field_aligned(G, ir, iz, mock_tok.gfile, phi0, decay=0.0, transits=1, substeps=1)
    assert M.shape == (G.grid.n_voxels, ir.size) and M.nnz > 0 and (M.data > 0).all()
    assert np.asarray(M.sum(axis=1)).max() <= 2.0 + 1e-9                                       # one pass per direction, each averaged
    sector = G.sector_of_voxel()
    on_plane = np.flatnonzero(sector == 3)
    sub = M[on_plane]                                                                         # on the source plane each pixel maps to itself
    hits = 0
    for p, (r, z) in enumerate(zip(ir, iz, strict=True)):
        v = G.eps_index(r, 3, z)
        row = M[v]
        if row.nnz and row[0, p] > 0:
            hits += 1
    assert hits >= 0.9 * ir.size and sub.nnz > 0
    M_decay = inv.mapping_field_aligned(G, ir, iz, mock_tok.gfile, phi0, decay=1.0, transits=1, substeps=1)
    assert M_decay.sum() < M.sum()                                                            # decay weakens the far sectors
