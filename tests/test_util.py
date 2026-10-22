"""
tests/test_util.py
==================
Unit tests for src/emis3d/Util.py
Covers coordinate conversions, wall helpers, JSON/H5 helpers, and utility functions.
No cherab or raysect objects are required.
"""

import os

import numpy as np
import pytest

from emis3d.util import (
    XY_To_RPhi,
    RPhi_To_XY,
    rpz_XYZ,
    xyz_RPZ,
    rz_to_xyz,
    length_along_wall,
    find_max_nested_lists,
    convert_arrays_to_list,
    split_revolutions,
    save_json,
    load_json,
    _ensure_path,
)

# ---------------------------------------------------------------------------
# XY_To_RPhi / RPhi_To_XY  round-trip
# ---------------------------------------------------------------------------


class TestXYRPhiRoundtrip:
    """XY -> (R, phi) -> XY should be lossless."""

    @pytest.mark.parametrize(
        "x,y",
        [
            (1.0, 0.0),  # on positive X-axis  → phi=0
            (0.0, 1.0),  # on positive Y-axis  → phi=π/2
            (-1.0, 0.0),  # on negative X-axis  → phi=π
            (0.0, -1.0),  # on negative Y-axis  → phi=-π/2
            (1.0, 1.0),  # first quadrant
            (2.0, -1.5),  # fourth quadrant, non-unit
        ],
    )
    def test_roundtrip(self, x, y):
        R, phi = XY_To_RPhi(x, y)
        x2, y2 = RPhi_To_XY(R, phi)
        assert np.isclose(x2, x, atol=1e-12)
        assert np.isclose(y2, y, atol=1e-12)

    def test_R_is_positive(self):
        for x, y in [(3.0, 4.0), (-3.0, 4.0), (-1.0, -1.0)]:
            R, _ = XY_To_RPhi(x, y)
            assert R >= 0.0

    def test_phi_range(self):
        """phi should always be in (-π, π]."""
        for x, y in [(1.0, 0.0), (-1.0, 0.5), (0.5, -2.0)]:
            _, phi = XY_To_RPhi(x, y)
            assert -np.pi <= phi <= np.pi, f"phi={phi:.4f} out of range for x={x}, y={y}"

    def test_toroidal_offset(self):
        """TorOffset shifts phi by the given amount (with wrapping)."""
        x, y = 1.0, 0.0
        _, phi_no_offset = XY_To_RPhi(x, y, TorOffset=0.0)
        _, phi_offset = XY_To_RPhi(x, y, TorOffset=np.pi / 4)
        assert np.isclose(phi_no_offset - np.pi / 4, phi_offset, atol=1e-12)


# ---------------------------------------------------------------------------
# rpz_XYZ  (forward cylindrical → Cartesian)
# ---------------------------------------------------------------------------


class TestRpzXYZ:
    def test_shape_preserved(self):
        rpz = np.array([[1.5, 2.0, 0.5], [0.0, np.pi / 2, np.pi], [0.0, 0.5, -0.3]])
        XYZ = rpz_XYZ(rpz)
        assert XYZ.shape == (3, 3)

    def test_on_x_axis(self):
        """R=2, phi=0, z=1 → x=2, y=0, z=1."""
        rpz = np.array([[2.0], [0.0], [1.0]])
        XYZ = rpz_XYZ(rpz)
        assert np.allclose(XYZ[:, 0], [2.0, 0.0, 1.0])

    def test_on_y_axis(self):
        """R=3, phi=π/2, z=0 → x≈0, y=3, z=0."""
        rpz = np.array([[3.0], [np.pi / 2], [0.0]])
        XYZ = rpz_XYZ(rpz)
        assert np.allclose(XYZ[:, 0], [0.0, 3.0, 0.0], atol=1e-12)

    def test_auto_transpose(self):
        """Function should accept (3,N) or (N,3) and return (3,N)."""
        # rpz_XYZ raises ValueError for non-3-leading arrays that can't be fixed
        rpz_ok = np.array([[1.0], [0.0], [0.0]])
        XYZ = rpz_XYZ(rpz_ok)
        assert XYZ.shape[0] == 3

    def test_roundtrip_with_xyz_RPZ(self):
        """rpz_XYZ followed by xyz_RPZ should recover the original R and z."""
        rpz_in = np.array([[1.5, 2.0], [0.3, 1.1], [-0.5, 0.7]])
        XYZ = rpz_XYZ(rpz_in)
        rpz_out = xyz_RPZ(XYZ)
        # R and z should be recovered exactly; phi may differ by sign convention bug
        assert np.allclose(rpz_out[0, :], rpz_in[0, :], atol=1e-12), "R not recovered"
        assert np.allclose(rpz_out[2, :], rpz_in[2, :], atol=1e-12), "z not recovered"


# ---------------------------------------------------------------------------
# xyz_RPZ
# ---------------------------------------------------------------------------


class TestXyzRPZPhi:

    def test_phi_argument_order_regression(self):
        """For x=2, y=1 the code returns arctan2(y,x)=26.6°."""
        XYZ = np.array([[2.0], [1.0], [0.0]])
        rpz = xyz_RPZ(XYZ)
        phi_deg = np.rad2deg(rpz[1, 0])
        # Current (buggy) value
        assert np.isclose(phi_deg, np.rad2deg(np.arctan2(1.0, 2.0)), atol=1e-6), "xyz_RPZ should return the standard aximuthal angle arctan2(y,x)"

    def test_roundtrip_phi(self):
        """rpiz_XYZ followed by xyz_RPZ must recover phi as well as R and z."""
        rpz_in = np.array([[1.5, 2.0], [0.3, 1.1], [-0.5, 0.7]])
        rpz_out = xyz_RPZ(rpz_XYZ(rpz_in))
        assert np.allclose(rpz_out, rpz_in, atol=1e-12)


# ---------------------------------------------------------------------------
# rz_to_xyz
# ---------------------------------------------------------------------------


class TestRzToXyz:
    def test_phi_zero(self):
        """At phi=0 the rotation is the identity: x=R, y=0."""
        R = np.array([1.5, 2.0])
        z = np.array([0.3, -0.1])
        x, y, z_out = rz_to_xyz(R, z, 0.0)
        assert np.allclose(x, R)
        assert np.allclose(y, np.zeros(2))
        assert np.allclose(z_out, z)

    def test_phi_pi_half(self):
        """At phi=π/2: x≈0, y=R."""
        R = np.array([2.0])
        z = np.array([0.5])
        x, y, _ = rz_to_xyz(R, z, np.pi / 2)
        assert np.allclose(x, [0.0], atol=1e-12)
        assert np.allclose(y, [2.0], atol=1e-12)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            rz_to_xyz(np.array([1.0, 2.0]), np.array([0.0]), 0.0)

    def test_returns_same_z(self):
        z = np.array([1.2, -3.4])
        _, _, z_out = rz_to_xyz(np.array([1.0, 2.0]), z, 1.0)
        assert np.array_equal(z_out, z)


# ---------------------------------------------------------------------------
# length_along_wall
# ---------------------------------------------------------------------------


class TestLengthAlongWall:
    def _unit_circle(self, N=100):
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.cos(theta), np.sin(theta)

    def test_total_length_circle(self):
        """Perimeter of a unit circle ≈ 2π."""
        R, Z = self._unit_circle(N=1000)
        _, Smax = length_along_wall(R, Z, R0=0.0)
        assert np.isclose(Smax, 2 * np.pi, rtol=1e-2)

    def test_Swall_all_nonneg(self):
        """All wall arc-lengths should be ≥ 0."""
        R, Z = self._unit_circle(N=200)
        S, _ = length_along_wall(R, Z, R0=0.0)
        assert np.all(S >= -1e-12)

    def test_Swall_length_equals_Nwall(self):
        R, Z = self._unit_circle(N=50)
        S, _ = length_along_wall(R, Z, R0=0.0)
        assert len(S) == 50


# ---------------------------------------------------------------------------
# find_max_nested_lists
# ---------------------------------------------------------------------------


class TestFindMaxNestedLists:
    def test_basic(self):
        assert find_max_nested_lists([[1, 5, 3], [2, 8, 4]]) == 8

    def test_single_inner_list(self):
        assert find_max_nested_lists([[7, 2]]) == 7

    def test_negative_values(self):
        assert find_max_nested_lists([[-3, -1], [-2, -4]]) == -1


# ---------------------------------------------------------------------------
# convert_arrays_to_list
# ---------------------------------------------------------------------------


class TestConvertArraysToList:
    def test_numpy_array(self):
        result = convert_arrays_to_list(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_nested_dict(self):
        obj = {"a": np.array([1.0, 2.0]), "b": {"c": np.array([3.0])}}
        result = convert_arrays_to_list(obj)
        assert result["a"] == [1.0, 2.0]
        assert result["b"]["c"] == [3.0]

    def test_list_of_arrays(self):
        obj = [np.array([1, 2]), np.array([3, 4])]
        result = convert_arrays_to_list(obj)
        assert result == [[1, 2], [3, 4]]

    def test_tuple_passthrough(self):
        obj = (np.array([1, 2]), 5)
        result = convert_arrays_to_list(obj)
        assert isinstance(result, tuple)
        assert result[0] == [1, 2]
        assert result[1] == 5

    def test_scalar_passthrough(self):
        assert convert_arrays_to_list(42) == 42
        assert convert_arrays_to_list("hello") == "hello"


# ---------------------------------------------------------------------------
# split_revolutions — including the off-by-one edge case
# ---------------------------------------------------------------------------


class TestSplitRevolutions:
    def _make_field_line(self, n_revs, N=300):
        phi = np.linspace(0, n_revs * 2 * np.pi, N, endpoint=False)
        x, y = np.cos(phi), np.sin(phi)
        z = np.zeros(N)
        R = np.ones(N)
        L = np.linspace(0, 10, N)
        return x, y, z, phi, R, L

    def test_one_revolution(self):
        revs = split_revolutions(*self._make_field_line(1))
        assert len(revs) == 1

    def test_two_revolutions(self):
        revs = split_revolutions(*self._make_field_line(2))
        assert len(revs) == 2

    def test_three_revolutions(self):
        revs = split_revolutions(*self._make_field_line(3))
        assert len(revs) == 3

    def test_phi_wrapped_to_0_2pi(self):
        """phi stored in each revolution should be in [0, 2π)."""
        revs = split_revolutions(*self._make_field_line(2))
        for r in revs:
            assert np.all(r["phi"] >= 0.0)
            assert np.all(r["phi"] < 2 * np.pi + 1e-10)

    def test_required_keys(self):
        revs = split_revolutions(*self._make_field_line(1))
        for key in ("x", "y", "z", "R", "L", "phi"):
            assert key in revs[0]

    @staticmethod
    def _traced(n_revs, direction, n_per_rev=100, phi0=None):
        """Like the tracer's output: endpoint included, phi increasing or decreasing."""
        phi0 = np.deg2rad(225.0) if phi0 is None else phi0
        phi = phi0 + direction * np.linspace(0, n_revs * 2 * np.pi, n_revs * n_per_rev + 1)
        L = np.linspace(0, 1, phi.size)                      # distance along the line, from the start
        return np.cos(phi), np.sin(phi), np.zeros(phi.size), phi, np.ones(phi.size), L

    @pytest.mark.parametrize("direction", [+1, -1])
    def test_both_directions_split_the_same_way(self, direction):
        """Each revolution starts with its own start sample, in travel order (also clockwise)."""
        revs = split_revolutions(*self._traced(3, direction))
        assert [len(r["L"]) for r in revs] == [100, 100, 100]
        for k, r in enumerate(revs):
            assert r["L"][0] == pytest.approx(k / 3)        # starts k turns along the line
            assert np.isclose(r["phi"][0], np.deg2rad(225.0))  # at the start angle
        assert np.all(np.diff(np.concatenate([r["L"] for r in revs])) > 0)

    def test_clockwise_first_revolution_holds_the_start(self):
        """It used to drop the start sample and hold the end of the line instead."""
        revs = split_revolutions(*self._traced(10, -1))
        assert len(revs) == 10
        assert revs[0]["L"][0] == 0.0 and revs[-1]["L"][-1] < 1.0


# ---------------------------------------------------------------------------
# save_json / load_json
# ---------------------------------------------------------------------------


class TestJsonHelpers:
    def test_roundtrip(self, tmp_path):
        data = {"a": [1, 2, 3], "b": {"c": 42.0}}
        save_json(data, str(tmp_path), "test.json")
        loaded = load_json(str(tmp_path / "test.json"))
        assert loaded == data

    def test_creates_directory(self, tmp_path):
        new_dir = str(tmp_path / "new" / "nested")
        save_json({"x": 1}, new_dir, "out.json")
        assert os.path.isfile(os.path.join(new_dir, "out.json"))


# ---------------------------------------------------------------------------
# _ensure_path  (nested-dict builder)
# ---------------------------------------------------------------------------


class TestEnsurePath:
    def test_creates_nested_key(self):
        data = {}
        _ensure_path(data, ["a", "b", "c"], default=99)
        assert data["a"]["b"]["c"] == 99

    def test_existing_value_preserved(self):
        data = {"a": {"b": 5}}
        _ensure_path(data, ["a", "b"], default=99)
        assert data["a"]["b"] == 5

    def test_default_func_applied(self):
        data = {}
        _ensure_path(data, ["x"], default=10, default_func=lambda v: v * 2)
        assert data["x"] == 20
