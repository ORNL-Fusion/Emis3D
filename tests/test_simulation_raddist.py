"""
tests/test_simulation_raddist.py

NIMROD and M3D-C1 radDists on a periodic phi grid (review issue 17).
Simulations usually write phi on [0, 2*pi) without the endpoint. Before the
fix, any chord crossing the last interval made the interpolator raise inside
Cherab, and bolos_observe stored the whole channel as 0.

Uses the built-in MOCK machine; its WRAP camera's chords cross phi = 0.
"""

import json

import numpy as np
import pytest

from emis3d.radDist import M3DC1, NIMROD

R0, Z0, SIG, SIG_PHI = 1.75, 0.15, 0.25, np.deg2rad(25.0)
CONFIG = {"tokamakName": "MOCK", "eqFileName": None, "units": "Brightness", "BOLOMETER_PROPS": {"pixelSamples": 20, "seed": 1}}
NPHI = 72  # 5-degree planes


def blob(R, z, phi):
    """A blob centred on phi = 0, so it straddles the seam of the grid."""
    dphi = (phi + np.pi) % (2 * np.pi) - np.pi
    return 1.0e6 * np.exp(-((R - R0) ** 2 + (z - Z0) ** 2) / (2 * SIG**2) - dphi**2 / (2 * SIG_PHI**2))


def phi_grid(kind):
    if kind == "open":  # [0, 2pi) without the endpoint, as simulations write it
        return np.linspace(0, 2 * np.pi, NPHI, endpoint=False)
    if kind == "closed":  # [0, 2pi]
        return np.linspace(0, 2 * np.pi, NPHI + 1)
    if kind == "centred":  # [-pi, pi)
        return np.linspace(-np.pi, np.pi, NPHI, endpoint=False)
    raise ValueError(kind)


def write_grid(path, axes_order=("z", "r", "phi"), kind="open", phi=None):
    r = np.linspace(1.05, 2.35, 53)
    z = np.linspace(-1.05, 1.05, 85)
    phi = phi_grid(kind) if phi is None else phi
    grids = dict(zip(axes_order, np.meshgrid(*({"r": r, "z": z, "phi": phi}[a] for a in axes_order), indexing="ij"), strict=True))
    values = blob(grids["r"], grids["z"], grids["phi"])
    path.write_text(json.dumps({"pradArray": values.tolist(), "r_list": r.tolist(), "z_list": z.tolist(), "phi_list": phi.tolist()}))
    return path


@pytest.fixture(scope="module")
def sims(tmp_path_factory):
    d = tmp_path_factory.mktemp("sim")
    out = {kind: NIMROD(nimrodFile=write_grid(d / f"n_{kind}.json", kind=kind), timestep=0, config=dict(CONFIG)) for kind in ("open", "closed", "centred")}
    out["m3dc1"] = M3DC1(m3dc1File=write_grid(d / "m.json", ("phi", "r", "z")), timestep=0, config=dict(CONFIG))
    for s in out.values():
        s.emissionName = s.info["emissionNames"][0]
    return out


def ev(sim, R, z, phi):
    return sim._evaluate(np.atleast_1d(R), np.atleast_1d(z), np.atleast_1d(phi))[sim.info["emissionNames"][0]]


def test_last_interval_of_an_open_grid_is_covered(sims):
    """phi between the last plane (355 deg) and 2pi used to raise inside the interpolator."""
    for deg in (355.0, 357.5, 359.9, 360.0, 362.5, -2.5):
        v = ev(sims["open"], R0, Z0, np.deg2rad(deg))[0]
        assert v == pytest.approx(blob(R0, Z0, np.deg2rad(deg)), rel=0.01)


def test_grid_conventions_give_identical_emissivity(sims):
    rng = np.random.default_rng(0)
    R = rng.uniform(1.1, 2.3, 400)
    z = rng.uniform(-1.0, 1.0, 400)
    phi = rng.uniform(-3 * np.pi, 3 * np.pi, 400)  # any angle, including far outside [0, 2pi)
    ref = ev(sims["closed"], R, z, phi)
    assert np.any(ref > 0)
    for kind in ("open", "centred", "m3dc1"):
        assert np.allclose(ev(sims[kind], R, z, phi), ref, rtol=1e-12, atol=0.0), kind


def test_single_phi_value_broadcasts(sims):
    R = np.array([1.6, 1.75, 1.9])
    one = ev(sims["open"], R, np.full(3, Z0), np.array([0.1]))
    each = ev(sims["open"], R, np.full(3, Z0), np.full(3, 0.1))
    assert np.array_equal(one, each)


def test_outside_the_grid_is_zero(sims):
    assert ev(sims["open"], 0.5, 0.0, 0.0)[0] == 0.0
    assert ev(sims["open"], R0, 2.0, 0.0)[0] == 0.0


def test_wrong_array_shape_is_rejected(tmp_path):
    f = write_grid(tmp_path / "x.json", axes_order=("r", "z", "phi"))  # saved in the wrong axis order
    with pytest.raises(ValueError, match="axis order"):
        NIMROD(nimrodFile=f, timestep=0, config=dict(CONFIG))


@pytest.mark.parametrize(
    "phi, match",
    [
        (np.linspace(0, np.pi, 37), "full turn is"),  # a sector, not a full turn
        (np.linspace(0, 3 * np.pi, 109), "more than a full turn"),
    ],
)
def test_incomplete_or_overlapping_phi_grid_is_rejected(tmp_path, phi, match):
    f = write_grid(tmp_path / "x.json", phi=phi)
    with pytest.raises(ValueError, match=match):
        NIMROD(nimrodFile=f, timestep=0, config=dict(CONFIG))


def test_failed_observation_raises_instead_of_storing_zero(sims, monkeypatch):
    sim = sims["closed"]

    def broken(*args, **kwargs):
        raise ValueError("interpolator out of range")

    monkeypatch.setattr(sim, "_evaluate", broken)
    monkeypatch.setitem(sim.info, "rayMethod", "python")  # so Cherab calls the (broken) Python function
    with pytest.raises(RuntimeError, match="Observation failed for foil .* of camera"):
        sim.bolos_observe()


@pytest.mark.cherab
def test_wrap_camera_sees_the_blob_on_an_open_grid(sims):
    """Every WRAP channel crosses phi = 0; on an open grid all were stored as 0 before."""
    signals = {}
    for kind in ("open", "closed"):
        sim = sims[kind]
        sim.data = {}
        sim.bolos_observe()
        signals[kind] = np.array([v[0] for v in sim.data["Brightness"]["NIMROD"]["WRAP"]])
    assert np.all(signals["open"] > 0)
    assert np.allclose(signals["open"], signals["closed"], rtol=1e-9)


def test_single_point_from_cherab(sims):
    """Cherab evaluates one point at a time through _evaluate_cherab. Under numpy 2
    the old code raised TypeError here (a 1-element array passed as a scalar), so
    every channel of every simulation radDist was stored as 0."""
    for sim in sims.values():
        v = sim._evaluate_cherab(R0, 0.0, Z0)  # x = R0, y = 0 -> phi = 0, the blob centre
        assert isinstance(v, float)
        assert v == pytest.approx(blob(R0, Z0, 0.0), rel=0.01)
