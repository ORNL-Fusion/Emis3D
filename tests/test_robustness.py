"""
tests/test_robustness.py

Helical emissivity with one angle per point, and errors that are reported
clearly instead of being swallowed.
"""

import tempfile

import numpy as np
import pytest

from emis3d import mock
from emis3d.radDist import Helical, HelicalRing
from emis3d.tokamak import Tokamak
from emis3d.util import config_loader
from emis3d.workspace import Workspace, set_workspace

BASE = {"tokamakName": "MOCK", "units": "Brightness", "injectionLocation": 225.0, "saveRunsDirectoryName": "p",
        "BOLOMETER_PROPS": {"pixelSamples": 10}, "eqFileName": mock.EQ_FILE, "numTransists": 1.0}


@pytest.fixture(scope="module")
def helical():
    set_workspace(Workspace.resolve(workspace=tempfile.mkdtemp(), env={}, user_config=tempfile.mktemp()))
    try:
        yield (Helical(startR=1.95, startZ=0.1, config={**BASE, "distType": "Helical", "sigma_R": 0.05,
                                                         "sigmaKernel": 0.03, "numFieldLines": 50}),
               HelicalRing(startR=1.95, startZ=0.1, config={**BASE, "distType": "HelicalRing", "sigma_R": 0.05, "sigma_z": 0.05}))
    finally:
        set_workspace(None)


@pytest.mark.parametrize("which", [0, 1], ids=["helical", "helicalring"])
def test_one_angle_per_point_matches_one_point_at_a_time(helical, which):
    """Helical used to mix field lines and angles when given several angles."""
    rd = helical[which]
    rng = np.random.default_rng(0)
    R, Z, P = rng.uniform(1.4, 2.1, 100), rng.uniform(-0.5, 0.5, 100), rng.uniform(0, 2 * np.pi, 100)
    for name in rd.info["emissionNames"]:
        together = rd.calc_emissivity(R, Z, phi=P, emissionName=name)[name]
        alone = [rd.calc_emissivity(R[i:i + 1], Z[i:i + 1], phi=P[i:i + 1], emissionName=name)[name][0] for i in range(R.size)]
        np.testing.assert_allclose(together, alone, rtol=1e-12, atol=0)


def test_helical_rejects_a_mismatched_angle_array(helical):
    with pytest.raises(ValueError, match="one per point"):
        helical[0].calc_emissivity(np.ones(3), np.zeros(3), phi=np.zeros(2), emissionName="counterClock_rev0")


def test_config_loader_names_the_file(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("key: [unclosed\n")
    with pytest.raises(ValueError, match="bad.yaml"):
        config_loader(bad)
    with pytest.raises(FileNotFoundError):
        config_loader(tmp_path / "missing.yaml")


def test_unreadable_equilibrium_is_an_error(tmp_path):
    folder = tmp_path / "equilibria" / "MOCK"
    folder.mkdir(parents=True)
    (folder / "broken.geqdsk").write_text("this is not a g-file\n")
    ws = Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml")
    with pytest.raises(RuntimeError, match="broken.geqdsk"):
        Tokamak(tokamakName="MOCK", mode="Analysis", eqFileName="broken.geqdsk", workspace=ws)


def test_missing_equilibrium_is_still_only_logged(tmp_path):
    """Rings do not need one, so an absent file is not an error (an unreadable one is)."""
    ws = Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml")
    assert Tokamak(tokamakName="MOCK", mode="Analysis", eqFileName="absent.geqdsk", workspace=ws).gfile is None
