"""
tests/test_fieldline_tracer.py

Field-line tracing on a circular test equilibrium with an exact safety factor.

    psi = B0 [(R - R0)^2 + Z^2] / 2   ->   B_R = -B0 Z / R,  B_Z = B0 (R - R0) / R
    B_phi = F(psi) / R

Every flux surface is a circle of radius r about (R0, 0), on which
q(r) = F / (B0 sqrt(R0^2 - r^2)) exactly. The equilibrium is written as a real
g-file and read back with freeqdsk, as Tokamak does.
"""

import io

import numpy as np
import pytest
from freeqdsk import geqdsk
from scipy.optimize import brentq

from emis3d.fieldline_tracer import Fieldline_Tracer

R0, A, B0, F0 = 1.70, 0.60, 1.0, 3.2


def F_of_r(r, eps):
    """F(psi) on the surface of radius r: F0 at the edge, (1 + eps) F0 on axis."""
    return F0 * (1 + eps * (1 - (r / A) ** 2))


def q_exact(r, eps):
    return F_of_r(r, eps) / (B0 * np.sqrt(R0**2 - r**2))


def make_gfile(eps=0.0, rcentr=1.60, n=129):
    Rg = np.linspace(R0 - 0.75, R0 + 0.75, n)
    Zg = np.linspace(-0.75, 0.75, n)
    RR, ZZ = np.meshgrid(Rg, Zg, indexing="ij")
    psin = np.linspace(0, 1, n)
    fpol = F0 * (1 + eps * (1 - psin))
    th = np.linspace(0, 2 * np.pi, 65)
    d = dict(
        nx=n,
        ny=n,
        rdim=Rg[-1] - Rg[0],
        zdim=Zg[-1] - Zg[0],
        rcentr=rcentr,
        rleft=Rg[0],
        zmid=0.0,
        rmagx=R0,
        zmagx=0.0,
        simagx=0.0,
        sibdry=B0 * A**2 / 2.0,
        bcentr=fpol[-1] / rcentr,
        cpasma=1e6,
        fpol=fpol,
        pres=np.zeros(n),
        ffprime=np.zeros(n),
        pprime=np.zeros(n),
        psi=B0 * ((RR - R0) ** 2 + ZZ**2) / 2.0,
        qpsi=fpol / (B0 * np.sqrt(R0**2 - A**2 * psin)),
        nbdry=65,
        nlim=65,
        rbdry=R0 + A * np.cos(th),
        zbdry=A * np.sin(th),
        rlim=R0 + 0.7 * np.cos(th),
        zlim=0.7 * np.sin(th),
    )
    buf = io.StringIO()
    geqdsk.write(d, buf)
    buf.seek(0)
    return geqdsk.read(buf)


PHI0 = np.deg2rad(225.0)


def trace(g, r, transits, n_per_transit=500, phi0=PHI0):
    t = Fieldline_Tracer(StartR=R0 + r, StartZ=0.0, StartPhi=phi0, gfile=g, NumTor=int(n_per_transit * abs(transits)))
    t.trace(NumTransits=transits)
    return t


def q_measured(t):
    """Toroidal angle needed for a whole number of poloidal turns, over 2 pi."""
    th = np.unwrap(np.arctan2(t.data["z"], t.data["R"] - R0))
    th = np.abs(th - th[0])
    k = np.floor(th[-1] / (2 * np.pi))
    phi_k = np.interp(2 * np.pi * k, th, np.abs(t.data["phi"] - t.data["phi"][0]))
    return phi_k / (2 * np.pi * k)


@pytest.mark.parametrize("eps", [0.0, 0.05])
@pytest.mark.parametrize("direction", [2.0, -2.0])
def test_q2_field_line_closes_after_two_transits(eps, direction):
    g = make_gfile(eps)
    r2 = brentq(lambda r: q_exact(r, eps) - 2.0, 0.01, 0.59)
    t = trace(g, r2, direction)
    miss = np.hypot(t.data["R"][-1] - t.data["R"][0], t.data["z"][-1] - t.data["z"][0])
    assert miss < 1e-4  # 0.1 mm


@pytest.mark.parametrize("eps", [0.0, 0.05])
@pytest.mark.parametrize("r", [0.15, 0.30, 0.45, 0.55])
def test_safety_factor_matches_exact(eps, r):
    t = trace(make_gfile(eps), r, 10.0)
    assert q_measured(t) == pytest.approx(q_exact(r, eps), rel=1e-5)


def test_field_line_stays_on_its_flux_surface():
    t = trace(make_gfile(0.05), 0.40, 3.0)
    r = np.hypot(t.data["R"] - R0, t.data["z"])
    assert np.max(np.abs(r - 0.40)) < 1e-5


def test_b_phi_uses_fpol_not_the_vacuum_field_at_the_axis():
    """rcentr differs from rmagx and F varies with psi: B_phi must follow F(psi)/R."""
    g = make_gfile(eps=0.05, rcentr=1.60)
    t = trace(g, 0.3, 0.1)
    for r in (0.0, 0.3, 0.55):
        R = R0 + r
        assert t.B_phi(R, 0.0) == pytest.approx(-F_of_r(r, 0.05) / R, rel=1e-4)
    # outside the last closed flux surface: the vacuum value, fpol at the boundary
    assert t.B_phi(R0 + 0.7, 0.0) == pytest.approx(-F0 / (R0 + 0.7), rel=1e-6)


def test_output_layout_is_unchanged():
    t = trace(make_gfile(), 0.3, 1.0)
    for key in ("x", "y", "z", "R", "L", "phi", "phi_wrapped"):
        assert t.data[key].shape == (500,)
    assert np.allclose(np.hypot(t.data["x"], t.data["y"]), t.data["R"])
    assert np.all(np.diff(t.data["L"]) > 0) and t.data["L"][0] == 0.0
    assert t.data["phi_wrapped"].min() >= 0 and t.data["phi_wrapped"].max() < 2 * np.pi
