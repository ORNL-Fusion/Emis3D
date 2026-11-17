"""
tests/test_fieldline_batch.py

Tracing many field lines at once (fieldline_tracer.trace_many). The lines
are integrated as one system, so each step evaluates the field for all of
them in one call. It must give what Fieldline_Tracer gives line by line,
and everything built on it (Tokamak.set_fieldlines, FieldLineMap.build) must
be unchanged.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from emis3d import fieldline_tracer as FT
from emis3d import mock
from emis3d.fieldline_map import FieldLineMap
from emis3d.geometry import GeometryMatrix, GridSpec
from emis3d.tokamak import Tokamak
from emis3d.workspace import Workspace

N_TOR = 400


@pytest.fixture(scope="module")
def tok():
    ws = Workspace.resolve(env={}, user_config=Path(tempfile.mktemp()))
    return Tokamak(tokamakName="MOCK", mode="Analysis", eqFileName=mock.EQ_FILE, workspace=ws)


def starts(n, seed=0):
    rng = np.random.default_rng(seed)
    return 1.75 + 0.3 * rng.random(n), -0.25 + 0.5 * rng.random(n)


def one_by_one(gfile, R0, Z0, phi0, transits, **kw):
    out = []
    for r, z in zip(R0, Z0, strict=True):
        t = FT.Fieldline_Tracer(StartR=r, StartZ=z, StartPhi=phi0, gfile=gfile, NumTor=N_TOR, **kw)
        t.trace(NumTransits=transits)
        out.append((t.data["R"], t.data["z"]))
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


@pytest.mark.parametrize("transits", [1.0, -1.0, 2.5, -0.5])
def test_batch_equals_line_by_line(tok, transits):
    R0, Z0 = starts(40)
    phi0 = np.deg2rad(225.0)
    many = FT.trace_many(tok.gfile, R0, Z0, phi0, NumTor=N_TOR, NumTransits=transits)
    R_ref, Z_ref = one_by_one(tok.gfile, R0, Z0, phi0, transits)
    assert many["R"].shape == (40, N_TOR) and many["phi"][0] == pytest.approx(phi0)
    assert many["phi"][-1] == pytest.approx(phi0 + 2 * np.pi * transits)
    np.testing.assert_allclose(many["R"], R_ref, atol=2e-7, rtol=0)
    np.testing.assert_allclose(many["z"], Z_ref, atol=2e-7, rtol=0)
    np.testing.assert_array_equal(many["R"][:, 0], R0)                          # every line starts where it was put


def test_chunks_give_the_same_lines(tok):
    R0, Z0 = starts(23, seed=1)
    whole = FT.trace_many(tok.gfile, R0, Z0, 0.3, NumTor=N_TOR)
    pieces = FT.trace_many(tok.gfile, R0, Z0, 0.3, NumTor=N_TOR, chunk=5)        # 5 + 5 + 5 + 5 + 3
    np.testing.assert_allclose(pieces["R"], whole["R"], atol=2e-7, rtol=0)
    np.testing.assert_allclose(pieces["z"], whole["z"], atol=2e-7, rtol=0)


def test_mirrored_helicity_is_passed_on(tok):
    R0, Z0 = starts(6, seed=2)
    many = FT.trace_many(tok.gfile, R0, Z0, 0.0, NumTor=N_TOR, mirror_helicity=True)
    R_ref, Z_ref = one_by_one(tok.gfile, R0, Z0, 0.0, 1.0, mirror_helicity=True)
    np.testing.assert_allclose(many["z"], Z_ref, atol=2e-7, rtol=0)
    normal = FT.trace_many(tok.gfile, R0, Z0, 0.0, NumTor=N_TOR)
    assert np.max(np.abs(normal["z"] - many["z"])) > 1e-2                        # the other way round poloidally


def test_a_batch_that_cannot_be_integrated_is_traced_line_by_line(tok, monkeypatch):
    """One bad line must not hide behind the batch: its own failure is what is reported."""
    real = FT.solve_ivp
    calls = []

    def flaky(fun, span, y0, **kw):
        calls.append(len(y0) // 2)
        if len(y0) > 2:                                                          # a batch: say it failed
            return type("Sol", (), {"success": False, "y": np.empty((0, 0)), "message": "failed"})()
        return real(fun, span, y0, **kw)

    monkeypatch.setattr(FT, "solve_ivp", flaky)
    R0, Z0 = starts(4, seed=3)
    out = FT.trace_many(tok.gfile, R0, Z0, 0.0, NumTor=N_TOR)
    assert calls == [4, 1, 1, 1, 1]
    monkeypatch.setattr(FT, "solve_ivp", real)
    np.testing.assert_allclose(out["R"], FT.trace_many(tok.gfile, R0, Z0, 0.0, NumTor=N_TOR)["R"], atol=2e-7, rtol=0)

    def always_fails(fun, span, y0, **kw):
        return type("Sol", (), {"success": False, "y": np.empty((0, 0)), "message": "step size too small"})()

    monkeypatch.setattr(FT, "solve_ivp", always_fails)
    with pytest.raises(RuntimeError, match="Field-line trace from R=.* failed: step size too small"):
        FT.trace_many(tok.gfile, R0, Z0, 0.0, NumTor=N_TOR)


def test_inputs_are_checked(tok):
    with pytest.raises(ValueError, match="gfile is None"):
        FT.trace_many(None, [1.9], [0.0], 0.0, NumTor=10)
    with pytest.raises(ValueError, match="same length"):
        FT.trace_many(tok.gfile, [1.9, 2.0], [0.0], 0.0, NumTor=10)


def test_the_field_is_built_once_per_gfile(tok):
    assert FT.field_of(tok.gfile) is FT.field_of(tok.gfile)
    assert FT.field_of(tok.gfile, mirror_helicity=True) is not FT.field_of(tok.gfile)
    t = FT.Fieldline_Tracer(StartR=1.9, StartZ=0.0, StartPhi=0.0, gfile=tok.gfile, NumTor=50)
    t.trace()
    assert t.field is FT.field_of(tok.gfile) and t.B_phi(1.9, 0.0) == pytest.approx(float(t.field.B_phi(1.9, 0.0)))


def test_set_fieldlines_stores_the_same_lines(tok):
    R0, Z0 = starts(5, seed=4)
    phi0 = np.deg2rad(90.0)
    tok.set_fieldlines(startR=R0, startZ=Z0, startPhi=phi0, numTransists=1.0)
    fl = tok.fieldLines["90"]
    assert fl["directionNames"] == ["counterClock_rev0", "clockwise_rev0"]
    for name, sign in (("counterClock_rev0", 1.0), ("clockwise_rev0", -1.0)):
        n = fl[name]["R"].shape[1]
        for j in range(5):
            t = FT.Fieldline_Tracer(StartR=R0[j], StartZ=Z0[j], StartPhi=phi0, gfile=tok.gfile, NumTor=501)
            t.trace(NumTransits=sign)
            np.testing.assert_allclose(fl[name]["R"][j], t.data["R"][:n], atol=2e-7, rtol=0)
            np.testing.assert_allclose(fl[name]["z"][j], t.data["z"][:n], atol=2e-7, rtol=0)
            np.testing.assert_allclose(fl[name]["L"][j], t.data["L"][:n], atol=1e-5, rtol=0)


def small_matrix(tok, dR=0.2, n_phi=8):
    g = GridSpec.from_wall(tok.wall, {"dR": dR, "dz": dR, "n_phi": n_phi})
    RR, ZZ = np.meshgrid(g.r_centers, g.z_centers, indexing="ij")
    inside_rz = tok._inside_tokamak(np.column_stack([RR.ravel(), ZZ.ravel()])).reshape(g.n_r, g.n_z)
    inside = np.repeat(inside_rz[:, None, :], g.n_phi, axis=1).ravel()
    return GeometryMatrix(machine="MOCK", grid=g, rows=sp.csr_matrix((1, g.n_voxels)), cameras=["A"], channels=["A01"],
                          camera_of_row=np.zeros(1, int), etendues=np.ones(1), inside_wall=inside, samples=1)


def test_mapping_is_the_same_as_traced_line_by_line(tok, monkeypatch):
    G = small_matrix(tok)
    batched = FieldLineMap.build(G, tok.gfile, 0.3, "t", mock.EQ_FILE, transits=1, substeps=2)

    real = FT.trace_many
    monkeypatch.setattr(FT, "trace_many", lambda *a, **kw: real(*a, **{**kw, "chunk": 1}))   # one line per solve
    single = FieldLineMap.build(G, tok.gfile, 0.3, "t", mock.EQ_FILE, transits=1, substeps=2)

    assert batched.n_pixels == single.n_pixels and len(batched.passes) == len(single.passes) == 2
    src = np.random.default_rng(0).uniform(size=batched.n_pixels)
    for (d1, m1, M1), (d2, m2, M2) in zip(batched.passes, single.passes, strict=True):
        assert (d1, m1) == (d2, m2)
        assert abs(M1 - M2).sum() < 1e-3 * abs(M2).sum()                         # (a line on a voxel's edge may land either side)
    np.testing.assert_allclose(batched.apply(src, 0.4), single.apply(src, 0.4), atol=1e-2 * src.max())
