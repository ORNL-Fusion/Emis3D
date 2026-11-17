"""
tests/test_radiated_power_grid.py

The radiated-power integral P_pol = integral of emissivity x R dR dz
(RadDist._total_radiated_power), which the toroidal profile, the TPF and the
total power are built on. Its grid used to cover the whole wall with 200
points each way, about 1 cm apart on a 2 m vessel, whatever the width of
the structure: a ring 5 mm wide was integrated 18% low, by an amount that
depended on where it sat between the grid points. The grid now covers the
emission's own extent (_emission_box).

A normalized Gaussian well inside the wall integrates to its centre's R.
"""

import numpy as np
import pytest

from emis3d import mock
from emis3d.radDist import ElongatedRing, Helical, SquareTube

BASE = {"tokamakName": "MOCK", "units": "Brightness", "injectionLocation": 225.0, "eqFileName": None, "saveRunsDirectoryName": "x",
        "BOLOMETER_PROPS": {"pixelSamples": 1, "seed": 0}}


def ring(sigma_R, sigma_z, R0, z0, rot=0.0):
    return ElongatedRing(startR=R0, startZ=z0, config={**BASE, "distType": "ElongatedRing", "sigma_R": sigma_R, "sigma_z": sigma_z, "rotationAngle": rot})


@pytest.mark.parametrize("sigma", [0.2, 0.05, 0.01, 0.005, 0.002])
@pytest.mark.parametrize("R0, z0", [(1.70, 0.0), (1.7035, 0.0031), (1.9021, -0.2507)])
def test_a_ring_integrates_to_its_radius_whatever_its_width(sigma, R0, z0):
    if sigma >= 0.2 and R0 > 1.8:
        pytest.skip("this wide ring reaches the wall, which cuts it")
    P = ring(sigma, sigma, R0, z0)._total_radiated_power(emissionName="elongatedRing")["P_pol"]
    assert P[0] == pytest.approx(R0, rel=5e-3 if sigma >= 0.2 else 1e-4)        # (a 0.2 m ring loses 0.4% beyond the wall)
    assert np.all(P == P[0])                                                     # toroidally uniform


def test_a_tilted_thin_ring(tmp_path):
    P = ring(0.003, 0.15, 1.7035, 0.0031, rot=37.0)._total_radiated_power(emissionName="elongatedRing")["P_pol"][0]
    assert P == pytest.approx(1.7035, rel=5e-3)


def test_a_square_tube_is_its_area_times_its_radius():
    tube = SquareTube(startR=1.75, startZ=0.1, config={**BASE, "distType": "SquareTube", "dR": 0.004, "dz": 0.006})
    P = tube._total_radiated_power(emissionName="squareTube")["P_pol"][0]
    assert P == pytest.approx(1.75 * (2 * 0.004) * (2 * 0.006), rel=0.03)        # sharp edges: to within a grid cell


def test_a_helical_structure_integrates_to_its_field_lines_mean_radius():
    hel = Helical(startR=1.9, startZ=0.0, config={**BASE, "distType": "Helical", "eqFileName": mock.EQ_FILE, "sigma_R": 0.03, "sigmaKernel": 0.004,
                                                  "numFieldLines": 12, "numTransists": 1.0})
    for name in hel.info["emissionNames"]:
        out = hel._total_radiated_power(n_phi=8, emissionName=name)
        R_lines, _ = hel.tokamak.find_RZ_Fline("225.0", name, inputPhis=out["phi_array"])     # (lines, angles)
        np.testing.assert_allclose(out["P_pol"], R_lines.mean(axis=0), rtol=1e-3)
        assert out["P_total"] == pytest.approx(out["P_pol"].mean() * 2 * np.pi)


def test_total_power_is_the_mean_times_two_pi():
    out = ring(0.05, 0.08, 1.75, 0.1)._total_radiated_power(emissionName="elongatedRing")
    assert out["P_total"] == pytest.approx(1.75 * 2 * np.pi, rel=1e-4) and len(out["phi_array"]) == len(out["P_pol"]) == 100
