"""
tests/test_util_raddist.py
==========================
Unit tests for src/emis3d/Util_radDist.py
Covers bivariate_normal, bivariate_normal_elongated, createRZGrid, and
bivariate_normal_isodensity_points.
No cherab, raysect, or file I/O required.
"""

import numpy as np
import pytest
from scipy.integrate import simpson as simps

from emis3d.util_radDist import (
    bivariate_normal,
    bivariate_normal_elongated,
    createRZGrid,
    bivariate_normal_isodensity_points,
)

# ---------------------------------------------------------------------------
# bivariate_normal
# ---------------------------------------------------------------------------


class TestBivariateNormal:

    def _integrate(self, fn_vals, R_vals, z_vals):
        """2-D numerical integral over a regular grid."""
        N = len(R_vals)
        grid = fn_vals.reshape(N, N)
        return simps(simps(grid, x=z_vals, axis=1), x=R_vals)

    def test_normalisation(self):
        """Integral over all space should be ~1."""
        R = np.linspace(-4, 4, 400)
        z = np.linspace(-4, 4, 400)
        RR, ZZ = np.meshgrid(R, z, indexing="ij")
        vals = bivariate_normal(RR.ravel(), ZZ.ravel(), R0=0.0, z0=0.0, sigma_z=0.5)
        integral = self._integrate(vals, R, z)
        assert np.isclose(integral, 1.0, atol=1e-3), f"integral = {integral:.6f}"

    def test_peak_at_centre(self):
        """Maximum should be at (R0, z0)."""
        R = np.linspace(0, 4, 200)
        z = np.linspace(-2, 2, 200)
        RR, ZZ = np.meshgrid(R, z, indexing="ij")
        vals = bivariate_normal(RR.ravel(), ZZ.ravel(), R0=2.0, z0=0.0, sigma_z=0.3)
        max_idx = np.argmax(vals)
        R_flat = RR.ravel()
        z_flat = ZZ.ravel()
        assert np.isclose(R_flat[max_idx], 2.0, atol=0.05)
        assert np.isclose(z_flat[max_idx], 0.0, atol=0.05)

    def test_symmetry(self):
        """bivariate_normal should be symmetric in R and z around (R0, z0)."""
        R0, z0, sig = 2.0, 1.0, 0.4
        for delta in [0.1, 0.3, 0.7]:
            v_plus = bivariate_normal(np.array([R0 + delta]), np.array([z0]), R0=R0, z0=z0, sigma_z=sig)
            v_minus = bivariate_normal(np.array([R0 - delta]), np.array([z0]), R0=R0, z0=z0, sigma_z=sig)
            assert np.isclose(v_plus[0], v_minus[0], atol=1e-12)

    def test_positive_everywhere(self):
        R = np.linspace(0, 5, 50)
        z = np.linspace(-3, 3, 50)
        RR, ZZ = np.meshgrid(R, z)
        vals = bivariate_normal(RR.ravel(), ZZ.ravel(), R0=2.0, z0=0.0, sigma_z=0.5)
        assert np.all(vals >= 0)

    def test_array_input(self):
        """Accepts 1-D arrays."""
        R = np.array([1.0, 2.0, 3.0])
        z = np.array([0.0, 0.0, 0.0])
        vals = bivariate_normal(R, z, R0=2.0, z0=0.0, sigma_z=1.0)
        assert vals.shape == (3,)


# ---------------------------------------------------------------------------
# bivariate_normal_elongated
# ---------------------------------------------------------------------------


class TestBivariateNormalElongated:

    def test_reduces_to_isotropic_at_sigma_R_1_theta_0(self):
        """sigma_R=1, theta=0 should equal bivariate_normal."""
        R = np.linspace(-2, 2, 100) + 2.0
        z = np.zeros_like(R)
        R0, z0, sig = 2.0, 0.0, 0.5

        v_elongated = bivariate_normal_elongated(R, z, R0=R0, z0=z0, sigma_R=sig, sigma_z=sig, theta=0.0)
        v_isotropic = bivariate_normal(R, z, R0=R0, z0=z0, sigma_z=sig)
        assert np.allclose(v_elongated, v_isotropic, atol=1e-8)

    def test_peak_at_centre(self):
        R = np.linspace(0, 4, 300)
        z = np.linspace(-2, 2, 300)
        RR, ZZ = np.meshgrid(R, z, indexing="ij")
        R0, z0 = 2.0, 0.5
        vals = bivariate_normal_elongated(RR.ravel(), ZZ.ravel(), R0=R0, z0=z0, sigma_R=0.8, sigma_z=0.3)
        idx = np.argmax(vals)
        assert np.isclose(RR.ravel()[idx], R0, atol=0.05)
        assert np.isclose(ZZ.ravel()[idx], z0, atol=0.05)

    def test_positive_everywhere(self):
        R = np.linspace(0, 5, 30)
        z = np.linspace(-3, 3, 30)
        RR, ZZ = np.meshgrid(R, z)
        vals = bivariate_normal_elongated(RR.ravel(), ZZ.ravel(), R0=2.0, z0=0.0, sigma_R=1.5, sigma_z=0.4)
        assert np.all(vals >= 0)

    def test_theta_changes_orientation(self):
        """Rotating 90° should swap the R and z sigma_R directions."""
        R0, z0, sig, elong = 2.0, 0.0, 0.2, 0.8
        # Point displaced in R direction
        R_disp = np.array([R0 + 0.3])
        z_mid = np.array([z0])
        # Point displaced in z direction
        R_mid = np.array([R0])
        z_disp = np.array([z0 + 0.3])

        v_R_0deg = bivariate_normal_elongated(R_disp, z_mid, R0=R0, z0=z0, sigma_R=elong, sigma_z=sig, theta=0.0)
        v_z_0deg = bivariate_normal_elongated(R_mid, z_disp, R0=R0, z0=z0, sigma_R=elong, sigma_z=sig, theta=0.0)
        v_R_90deg = bivariate_normal_elongated(R_disp, z_mid, R0=R0, z0=z0, sigma_R=elong, sigma_z=sig, theta=90.0)
        v_z_90deg = bivariate_normal_elongated(R_mid, z_disp, R0=R0, z0=z0, sigma_R=elong, sigma_z=sig, theta=90.0)
        # At theta=0: elong > sig so the R direction is broader, giving higher value
        # At theta=90: the roles swap
        assert v_R_0deg[0] > v_z_0deg[0]
        assert v_z_90deg[0] > v_R_90deg[0]


# ---------------------------------------------------------------------------
# createRZGrid
# ---------------------------------------------------------------------------


class TestCreateRZGrid:

    def test_shape_no_wall(self):
        rz = createRZGrid((1.0, 2.5), (-1.0, 1.0), num_r=10, num_z=8)
        assert rz.shape == (80, 2)

    def test_r_bounds(self):
        r_min, r_max = 1.2, 2.8
        rz = createRZGrid((r_min, r_max), (-0.5, 0.5), num_r=20, num_z=5)
        assert rz[:, 0].min() >= r_min - 1e-10
        assert rz[:, 0].max() <= r_max + 1e-10

    def test_z_bounds(self):
        z_min, z_max = -1.5, 1.5
        rz = createRZGrid((1.0, 2.0), (z_min, z_max), num_r=5, num_z=20)
        assert rz[:, 1].min() >= z_min - 1e-10
        assert rz[:, 1].max() <= z_max + 1e-10

    def test_wall_mask_reduces_points(self):
        """When a wall curve is provided, points outside it are dropped."""
        from matplotlib.path import Path

        # Square wall: [1,2] × [-0.5, 0.5]
        verts = [(1.0, -0.5), (2.0, -0.5), (2.0, 0.5), (1.0, 0.5), (1.0, -0.5)]
        wall = Path(verts)
        rz_all = createRZGrid((0.5, 2.5), (-1.0, 1.0), num_r=20, num_z=20)
        rz_wall = createRZGrid((0.5, 2.5), (-1.0, 1.0), num_r=20, num_z=20, wallcurve=wall)
        assert rz_wall.shape[0] < rz_all.shape[0]

    def test_two_column_output(self):
        rz = createRZGrid((1.0, 2.0), (-1.0, 1.0), num_r=5, num_z=5)
        assert rz.shape[1] == 2


# ---------------------------------------------------------------------------
# bivariate_normal_isodensity_points
# ---------------------------------------------------------------------------


class TestIsodesityPoints:

    def test_output_shapes(self):
        points, weights = bivariate_normal_isodensity_points(R0=2.0, z0=0.0, sigma_target=0.5, sigma_kernel=0.1, n=30, seed=0)
        assert points.shape == (30, 2)
        assert weights.shape == (30,)

    def test_weights_sum_to_one(self):
        _, weights = bivariate_normal_isodensity_points(R0=1.0, z0=0.5, sigma_target=0.4, sigma_kernel=0.2, n=50, seed=7)
        assert np.isclose(weights.sum(), 1.0)

    def test_equal_weights(self):
        _, weights = bivariate_normal_isodensity_points(R0=0.0, z0=0.0, sigma_target=0.3, sigma_kernel=0.1, n=20, seed=1)
        assert np.allclose(weights, weights[0])

    def test_raises_when_kernel_ge_target(self):
        with pytest.raises(ValueError, match="sigma_kernel"):
            bivariate_normal_isodensity_points(R0=0.0, z0=0.0, sigma_target=0.1, sigma_kernel=0.2, n=10)

    def test_raises_when_kernel_equals_target(self):
        with pytest.raises(ValueError):
            bivariate_normal_isodensity_points(R0=0.0, z0=0.0, sigma_target=0.3, sigma_kernel=0.3, n=10)

    def test_reproducible_with_seed(self):
        pts1, _ = bivariate_normal_isodensity_points(0.0, 0.0, 0.5, 0.1, n=20, seed=42)
        pts2, _ = bivariate_normal_isodensity_points(0.0, 0.0, 0.5, 0.1, n=20, seed=42)
        assert np.array_equal(pts1, pts2)

    def test_different_seeds_give_different_points(self):
        pts1, _ = bivariate_normal_isodensity_points(0.0, 0.0, 0.5, 0.1, n=20, seed=1)
        pts2, _ = bivariate_normal_isodensity_points(0.0, 0.0, 0.5, 0.1, n=20, seed=2)
        assert not np.array_equal(pts1, pts2)

    def test_points_centred_near_R0_z0(self):
        """The mean of source points should be close to (R0, z0)."""
        R0, z0 = 2.0, 0.5
        pts, _ = bivariate_normal_isodensity_points(R0, z0, 0.5, 0.1, n=2000, seed=99)
        assert np.isclose(pts[:, 0].mean(), R0, atol=0.05)
        assert np.isclose(pts[:, 1].mean(), z0, atol=0.05)

    def test_source_sigma_correct(self):
        """
        The source distribution std should equal sqrt(sigma_target² - sigma_kernel²).
        """
        sig_t, sig_k = 0.5, 0.3
        expected_sigma = np.sqrt(sig_t**2 - sig_k**2)
        pts, _ = bivariate_normal_isodensity_points(0.0, 0.0, sig_t, sig_k, n=5000, seed=0)
        measured_sigma = pts.std()
        assert np.isclose(measured_sigma, expected_sigma, rtol=0.1)
