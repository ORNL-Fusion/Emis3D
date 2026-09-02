"""
tests/test_elongated_ring_observation.py
========================================
Integration / regression test for the Cherab observation path.

Builds a single, fully-specified ElongatedRing radiation distribution for
DIII-D, observes it with Cherab (``bolos_observe`` -> ``foil.observe()``), and
checks the per-channel brightness on one bolometer group against a stored
golden result (tests/data/elongated_ring_brightness_golden.json).

Unlike the helical distribution, ElongatedRing needs no magnetic-equilibrium
g-file, so this test only depends on the in-repo DIII-D tokamak geometry.

WHY THIS IS OPT-IN
------------------
``foil.observe()`` ray-traces with ``pixelSamples`` rays per channel, so a full
build + observe takes ~35 s and the result is *stochastic*.  To keep the default
``python -m pytest tests/`` run fast and deterministic, this test is skipped
unless the environment variable ``EMIS3D_RUN_CHERAB`` is set:

    EMIS3D_RUN_CHERAB=1 python -m pytest tests/test_elongated_ring_observation.py -v

Monte-Carlo scatter is absorbed by a combined tolerance: a relative ``rtol`` for
the bright channels plus an absolute floor (a small fraction of the peak
brightness) so the noisy near-zero tail channels do not cause spurious failures.
Both values live in the golden file's ``_meta`` block.

REGENERATING THE GOLDEN
-----------------------
If the observation physics changes intentionally, regenerate the baseline with
``scripts``/the generator used to create it (build the same fixture, capture
``rD.data[units]["elongatedRing"][BOLO_GROUP]``, and overwrite the JSON).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

GOLDEN_PATH = Path(__file__).parent / "data" / "elongated_ring_brightness_golden.json"

# Opt-in guard: only run when explicitly requested (slow + stochastic).
pytestmark = pytest.mark.skipif(
    not os.environ.get("EMIS3D_RUN_CHERAB"),
    reason="Cherab observation test is slow/stochastic; set EMIS3D_RUN_CHERAB=1 to run.",
)


def _load_golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def _build_and_observe(meta):
    """Build the pinned ElongatedRing and observe a single bolometer group.

    Returns the per-channel brightness (np.ndarray) for ``meta['boloGroup']``.
    Calls calc_radiated_power + bolos_observe directly (not ``build()``) to avoid
    the saveRadDist disk write into the git-ignored inputs/ directory.
    """
    import main.radDist as radDist

    rD = radDist.ElongatedRing(
        startR=meta["startR"], startZ=meta["startZ"], config=dict(meta["config"])
    )

    # Observe only the group the golden was captured on, to keep runtime down.
    rD.tokamak.bolometers = [
        b for b in rD.tokamak.bolometers if b.name == meta["boloGroup"]
    ]
    assert rD.tokamak.bolometers, f"bolometer group {meta['boloGroup']!r} not found"

    rD.calc_radiated_power()
    rD.bolos_observe()

    units = rD.info["units"]
    return np.asarray(
        rD.data[units]["elongatedRing"][meta["boloGroup"]], dtype=float
    )


@pytest.fixture(scope="module")
def observed():
    """Build + observe once; shared across the assertions in this module."""
    pytest.importorskip("cherab", reason="cherab not installed")
    pytest.importorskip("raysect", reason="raysect not installed")
    golden = _load_golden()
    return golden, _build_and_observe(golden["_meta"])


class TestElongatedRingObservation:

    def test_channel_count_matches_golden(self, observed):
        golden, brightness = observed
        assert len(brightness) == len(golden["brightness"])

    def test_brightness_is_physical(self, observed):
        _, brightness = observed
        # Brightness [W/m^2] must be non-negative and not all-zero.
        assert np.all(brightness >= 0.0)
        assert brightness.max() > 0.0

    def test_brightness_matches_golden(self, observed):
        golden, brightness = observed
        expected = np.asarray(golden["brightness"], dtype=float)
        meta = golden["_meta"]

        rtol = meta["rtol"]
        atol = meta["atol_frac_of_peak"] * expected.max()

        close = np.isclose(brightness, expected, rtol=rtol, atol=atol)
        if not np.all(close):
            bad = np.where(~close)[0]
            lines = "\n".join(
                f"  ch {i}: got {brightness[i]:.4e}, expected {expected[i]:.4e}"
                for i in bad
            )
            pytest.fail(
                f"{len(bad)}/{len(expected)} channels outside tolerance "
                f"(rtol={rtol}, atol={atol:.3e}):\n{lines}"
            )
