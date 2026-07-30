"""
Regression tests for the Tokamak.fieldLines key mismatch.

The write side used f"{int(np.rad2deg(startPhiRad))}" and the read side used
str(info["startPhi"]); these disagreed whenever the deg->rad->deg round trip
truncated (240 -> "239") or when the config value was a float (45.0 ->
"45.0"). Both raised for every injection location except the few that happen
to round-trip exactly, e.g. 225.
"""

import numpy as np
import pytest

from main.Util import fieldline_key


class TestFieldlineKey:

    @pytest.mark.parametrize("deg", [0, 15, 45, 60, 105, 225, 240, 315, 359])
    def test_write_and_read_keys_agree(self, deg):
        """The core regression: radians in (write) must match degrees in (read)."""
        write = fieldline_key(np.deg2rad(deg), degrees=False)
        read = fieldline_key(deg)
        assert write == read == str(deg)

    @pytest.mark.parametrize("value", [45, 45.0, "45", "45.0", np.float64(45.0)])
    def test_type_insensitive(self, value):
        """int, float, str and numpy scalars all give the same key."""
        assert fieldline_key(value) == "45"

    def test_truncation_cases_that_used_to_break(self):
        """These are exactly the angles the old int() truncation corrupted."""
        for deg in (15, 60, 240):
            assert fieldline_key(np.deg2rad(deg), degrees=False) == str(deg)

    @pytest.mark.parametrize(
        "value,expected",
        [(405, "45"), (-30, "330"), (360, "0"), (359.6, "0"), (0.4, "0")],
    )
    def test_wrapping_and_rounding(self, value, expected):
        assert fieldline_key(value) == expected

    def test_radians_flag(self):
        assert fieldline_key(np.pi, degrees=False) == "180"
        assert fieldline_key(2 * np.pi, degrees=False) == "0"


class TestTokamakRoundTrip:
    """set_fieldlines(radians) followed by find_RZ_Fline(config degrees)."""

    @pytest.fixture
    def fake_tracing(self, monkeypatch):
        from main.Tokamak import Tokamak

        class FakeTracer:
            def __init__(self, StartR, StartZ, StartPhi, gfile, NumTor):
                self.R0, self.Z0 = StartR, StartZ

            def trace(self, NumTransits):
                n = 64
                phi = np.linspace(0, 2 * np.pi * np.sign(NumTransits), n)
                self.data = {
                    "x": np.full(n, self.R0),
                    "y": np.zeros(n),
                    "z": np.full(n, self.Z0),
                    "phi": phi,
                    "R": np.full(n, self.R0),
                    "L": np.linspace(0, 1, n),
                }

        g = Tokamak.set_fieldlines.__globals__
        monkeypatch.setitem(g, "Fieldline_Tracer", FakeTracer)
        monkeypatch.setitem(
            g,
            "split_revolutions",
            lambda x, y, z, phi, R, L: [
                {"x": x, "y": y, "z": z, "phi": phi, "R": R, "L": L}
            ],
        )
        return Tokamak

    @pytest.mark.parametrize("inj", [225, 240, 60, 15, 45.0])
    def test_trace_then_lookup(self, fake_tracing, inj):
        Tokamak = fake_tracing
        tok = Tokamak.__new__(Tokamak)
        tok.verbose = False
        tok.gfile = None

        tok.set_fieldlines(
            startR=[2.0],
            startZ=[0.0],
            startPhi=np.deg2rad(float(inj)),
            numTransists=1.0,
        )
        key = fieldline_key(inj)
        assert key in tok.fieldLines

        emis = tok.fieldLines[key]["directionNames"][0]
        # Look it up the way _evaluate does: str(info["startPhi"]) in degrees
        R, z = tok.find_RZ_Fline(str(inj), emis, inputPhis=np.array([0.5]))
        assert np.isclose(R.ravel()[0], 2.0)

    def test_unknown_angle_raises_informative_keyerror(self, fake_tracing):
        Tokamak = fake_tracing
        tok = Tokamak.__new__(Tokamak)
        tok.verbose = False
        tok.gfile = None
        tok.set_fieldlines(
            startR=[2.0],
            startZ=[0.0],
            startPhi=np.deg2rad(225.0),
            numTransists=1.0,
        )
        emis = tok.fieldLines["225"]["directionNames"][0]
        with pytest.raises(KeyError, match="No field lines traced"):
            tok.find_RZ_Fline("30", emis, inputPhis=np.array([0.5]))
