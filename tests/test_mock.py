"""
tests/test_mock.py

The built-in MOCK machine (src/emis3d/mock.py): the shipped files match the
generator, its cameras see what they are meant to see, and test variants work.
The full self-test ray-traces and is opt-in: `pytest -m cherab`.
"""

import filecmp

import numpy as np
import pytest

from emis3d import mock
from emis3d.workspace import BUILTIN_MACHINES, Workspace


@pytest.fixture(scope="module")
def mock_tokamak(tmp_path_factory):
    from emis3d.Tokamak import Tokamak

    ws = Workspace.resolve(env={}, user_config=tmp_path_factory.mktemp("u") / "none.yaml")
    return Tokamak(tokamakName="MOCK", mode="Analysis", loadBolometers=True, workspace=ws)


def chord(tok, foil, smax=5.0, n=5001):
    """Central sightline until it first leaves the plasma: (path length, phi in degrees)."""
    p0, d = foil.slit.centre_point, foil.sightline_vector.normalise()
    s = np.linspace(0.0, smax, n)
    x, y, z = p0.x + s * d.x, p0.y + s * d.y, p0.z + s * d.z
    R = np.hypot(x, y)
    inside = tok._inside_tokamak(np.column_stack([R, z]))
    if not inside.any():
        return 0.0, np.array([])
    first = int(np.argmax(inside))
    rest = ~inside[first:]
    stop = first + int(np.argmax(rest)) if rest.any() else n
    return s[stop - 1] - s[first], np.rad2deg(np.arctan2(y[first:stop], x[first:stop])) % 360.0


def test_shipped_files_match_the_generator(tmp_path):
    mock.build(tmp_path)
    cmp = filecmp.dircmp(tmp_path / "MOCK", BUILTIN_MACHINES / "MOCK")

    def differences(c):
        out = c.left_only + c.right_only + c.diff_files
        for sub in c.subdirs.values():
            out += differences(sub)
        return out

    assert differences(cmp) == [], "shipped MOCK is stale: run `python -m emis3d.mock`"


def test_wall_is_closed_and_volume_consistent():
    rz = mock.wall_polygon()
    assert np.allclose(rz[0], rz[-1])
    area, volume = mock.wall_area_and_volume(rz)
    assert area == pytest.approx(np.pi * 0.6 * 0.6 * 1.7, rel=0.05)       # ~ pi a (kappa a)
    assert volume == pytest.approx(mock.settings()["MACHINE"]["volume"], rel=1e-4)


def test_every_chord_crosses_the_plasma(mock_tokamak):
    for bolo in mock_tokamak.bolometers:
        for foil in bolo.bolometer_camera:
            length, _ = chord(mock_tokamak, foil)
            assert length > 0.5, f"{foil.name} misses the plasma"


def test_camera_toroidal_coverage(mock_tokamak):
    cams = {b.name: b for b in mock_tokamak.bolometers}
    for name, phi0 in (("POL_A", 45.0), ("POL_B", 165.0), ("POL_C", 285.0)):
        for foil in cams[name].bolometer_camera:
            _, phi = chord(mock_tokamak, foil)
            assert np.allclose(phi, phi0, atol=1e-6), f"{foil.name} is not poloidal"
    spans = [np.ptp(chord(mock_tokamak, f)[1]) for f in cams["SKEW"].bolometer_camera]
    assert max(spans) > 40.0                                  # toroidal + poloidal views
    for foil in cams["WRAP"].bolometer_camera:
        _, phi = chord(mock_tokamak, foil)
        assert phi.min() < 5.0 and phi.max() > 355.0, f"{foil.name} does not cross phi = 0"


def test_etendues_are_stored_and_positive(mock_tokamak):
    for bolo in mock_tokamak.bolometers:
        assert len(bolo.info["ETENDUE"]) == bolo.info["NUM_CHANNELS"] == len(bolo.info["CHANNEL_TAGS"])
        assert all(e > 0 for e in bolo.info["ETENDUE"])


def test_variant_with_two_toroidal_locations(tmp_path):
    from emis3d.Tokamak import Tokamak

    ws = mock.mock_workspace(tmp_path, cameras=[mock.CameraSpec("A", 90.0), mock.CameraSpec("B", 225.0)])
    tok = Tokamak(tokamakName="MOCK", mode="Analysis", loadBolometers=True, workspace=ws)
    assert sorted(b.name for b in tok.bolometers) == ["A", "B"]
    assert sorted(b.info["CAMERA_POSITION_R_Z_PHI"][2] for b in tok.bolometers) == [90.0, 225.0]


@pytest.mark.cherab
def test_selftest(tmp_path):
    summary = mock.selftest(workdir=tmp_path, pixel_samples=50, verbose=False)
    assert summary["chosen_radDist"] == summary["true_radDist"]
