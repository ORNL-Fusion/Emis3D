"""
tests/test_seeded_observation_reproducible.py
=============================================
Regression test for reproducible Cherab observations.

With an RNG seed set in ``BOLOMETER_PROPS``, an ``ElongatedRing`` observed twice
yields identical per-channel brightness. ElongatedRing needs no equilibrium
g-file, so this depends only on the in-repo DIII-D geometry.

Opt-in (slow): ``foil.observe()`` ray-traces, so this is deselected by default
via the ``cherab`` marker. Run it explicitly with:

    pytest -m cherab
"""

import numpy as np
import pytest

# `pytestmark` applies a marker to *every* test in this file. The `cherab` marker
# (registered in pyproject.toml) means "slow, needs cherab": these tests are
# deselected from a normal `pytest` run and execute only via `pytest -m cherab`.
pytestmark = pytest.mark.cherab

BOLO_GROUP = "SX90PF_UP"
CONFIG = {
    "distType": "ElongatedRing",
    "tokamakName": "DIII-D",
    "units": "Brightness",
    "injectionLocation": 225,
    "eqFileName": "",
    "sigma_R": 0.3,
    "sigma_z": 0.25,
    "rotationAngle": 30,
    "BOLOMETER_PROPS": {"pixelSamples": 200, "numProcessors": 1, "seed": 12345},
}


def test_seeded_observation_is_reproducible():
    """Two seeded observes of one structure produce bit-identical brightness."""
    pytest.importorskip("cherab", reason="cherab not installed")
    pytest.importorskip("raysect", reason="raysect not installed")
    import main.radDist as radDist

    rD = radDist.ElongatedRing(startR=2.055, startZ=0.414, config=dict(CONFIG))
    rD.tokamak.bolometers = [b for b in rD.tokamak.bolometers if b.name == BOLO_GROUP]

    def observe():
        rD.bolos_observe()
        return np.asarray(
            rD.data[rD.info["units"]]["elongatedRing"][BOLO_GROUP], dtype=float
        )

    first = observe()
    second = observe()

    assert first.max() > 0.0, "observation is empty; nothing meaningful to compare"
    np.testing.assert_array_equal(first, second)
