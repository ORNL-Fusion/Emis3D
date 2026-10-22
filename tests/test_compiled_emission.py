"""
tests/test_compiled_emission.py

Compiled emission (rays: {method: compiled}): Cherab evaluates a precomputed
grid with its own compiled interpolator instead of calling Python at every
step of every ray. Checked on MOCK against the Python path: point by point
for every radDist type, the choice of emitter per camera, the toroidal power
profile, and (ray tracing) the camera signals and the speed.
"""

import contextlib
import io
import json
import time

import numpy as np
import pytest

import emis3d.radDist as RD
from emis3d import mock
from emis3d.raddist_library import RadDistLibrary
from emis3d.workspace import Workspace, set_workspace

BASE = {"tokamakName": "MOCK", "units": "Brightness", "injectionLocation": 225.0, "saveRunsDirectoryName": "p",
        "BOLOMETER_PROPS": {"pixelSamples": 10, "seed": 0}, "integrationStep": 0.01}


@pytest.fixture(scope="module")
def raddists(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ws")
    set_workspace(Workspace.resolve(workspace=tmp, env={}, user_config=tmp / "none.yaml"))
    r, z, ph = np.linspace(1.05, 2.35, 27), np.linspace(-1.05, 1.05, 43), np.linspace(0, 2 * np.pi, 36, endpoint=False)
    Z, R, P = np.meshgrid(z, r, ph, indexing="ij")
    prad = np.exp(-((R - 1.85) ** 2 / 0.08 + (Z - 0.1) ** 2 / 0.125)) * (1 + 0.8 * np.cos(P - np.deg2rad(225)))
    (tmp / "sim.json").write_text(json.dumps({"pradArray": prad.tolist(), "r_list": r.tolist(), "z_list": z.tolist(), "phi_list": ph.tolist()}))
    with contextlib.redirect_stdout(io.StringIO()):
        out = {
            "ring": RD.ElongatedRing(startR=1.85, startZ=0.1, config={**BASE, "distType": "ElongatedRing", "eqFileName": None,
                                                                     "sigma_R": 0.15, "sigma_z": 0.2, "rotationAngle": 30.0}),
            "tube": RD.SquareTube(startR=1.85, startZ=0.1, config={**BASE, "distType": "SquareTube", "eqFileName": None, "dR": 0.15, "dz": 0.2}),
            "helical": RD.Helical(startR=1.95, startZ=0.1, config={**BASE, "distType": "Helical", "eqFileName": mock.EQ_FILE,
                                                                   "sigma_R": 0.05, "sigmaKernel": 0.02, "numFieldLines": 30, "numTransists": 1.0}),
            "helicalring": RD.HelicalRing(startR=1.95, startZ=0.1, config={**BASE, "distType": "HelicalRing", "eqFileName": mock.EQ_FILE,
                                                                           "sigma_R": 0.3, "sigma_z": 0.1, "numTransists": 1.0}),
            "nimrod": RD.NIMROD(nimrodFile=str(tmp / "sim.json"), timestep=1, config={**BASE, "eqFileName": None}),
        }
    yield out
    set_workspace(None)


@pytest.mark.parametrize("kind, mean_tol, max_tol", [
    ("ring", 1e-4, 2e-2), ("helical", 1e-4, 1e-2), ("helicalring", 1e-4, 1e-2),
    ("tube", 3e-3, 1.0),        # a sharp-edged step: blurred over one 1 mm cell, so single points can differ fully
    ("nimrod", 1e-3, 0.1),      # a coarse 5 cm simulation grid meeting the wall
])
def test_compiled_grid_matches_the_python_emissivity(raddists, kind, mean_tol, max_tol):
    rd = raddists[kind]
    rng = np.random.default_rng(1)
    plane = np.deg2rad(100.0)
    R, Z = rng.uniform(1.15, 2.25, 1500), rng.uniform(-0.8, 0.8, 1500)
    P = np.full(R.size, plane) if kind.startswith("helical") else rng.uniform(-np.pi, np.pi, R.size)
    for name in rd.info["emissionNames"]:
        f = rd._compiled_emission(name, plane)
        comp = np.array([f(a * np.cos(c), a * np.sin(c), b) for a, b, c in zip(R, Z, P, strict=True)])
        ref = np.concatenate([rd.calc_emissivity(R[i:i + 1], Z[i:i + 1], phi=P[i:i + 1], emissionName=name)[name] for i in range(R.size)])
        err = np.abs(comp - ref) / ref.max()
        assert err.mean() < mean_tol and err.max() <= max_tol


def test_each_camera_gets_the_right_emitter(raddists):
    hel = raddists["helical"]
    spans = {b.name: hel._camera_view_phi(b)[1] for b in hel.tokamak.bolometers}
    assert all(spans[c] < 0.5 for c in ("POL_A", "POL_B", "POL_C")) and spans["SKEW"] > 90 and spans["WRAP"] > 90
    how = {b.name: hel._emission_function("counterClock_rev0", b, {})[1] for b in hel.tokamak.bolometers}
    assert how == {"POL_A": "plane", "POL_B": "plane", "POL_C": "plane", "SKEW": "python", "WRAP": "python"}

    cache: dict = {}
    ring = raddists["ring"]
    kinds = {ring._emission_function("elongatedRing", b, cache)[1] for b in ring.tokamak.bolometers}
    assert kinds == {"uniform"} and len(cache) == 1                                      # one grid for every camera
    assert {raddists["nimrod"]._emission_function("NIMROD", b, {})[1] for b in ring.tokamak.bolometers} == {"grid3d"}


def test_python_method_still_available(raddists, monkeypatch):
    ring = raddists["ring"]
    monkeypatch.setitem(ring.info, "rayMethod", "python")
    fn, how = ring._emission_function("elongatedRing", ring.tokamak.bolometers[0], {})
    assert how == "python" and fn == ring._evaluate_cherab


def test_emission_outside_its_box_is_zero(raddists):
    hel = raddists["helical"]
    f = hel._compiled_emission("counterClock_rev0", np.deg2rad(100.0))
    assert f(-1.0, 1e-9, 0.0) == 0.0 and f(0.0, 2.2, 0.8) == 0.0   # far from the field lines, in that plane


def test_uniform_power_profile_is_computed_once_and_unchanged(raddists, monkeypatch):
    ring = raddists["ring"]
    once = ring._total_radiated_power(n_phi=12, n_poloidal=60, emissionName="elongatedRing")
    monkeypatch.setattr(type(ring), "TOROIDALLY_UNIFORM", False)
    every = ring._total_radiated_power(n_phi=12, n_poloidal=60, emissionName="elongatedRing")
    np.testing.assert_allclose(once["P_pol"], every["P_pol"], rtol=1e-12)


def test_grid_spacing_follows_the_narrowest_width(raddists, monkeypatch):
    assert raddists["helical"]._grid_spacing() == pytest.approx(0.02 / 8)
    assert raddists["ring"]._grid_spacing() == pytest.approx(3e-3)                       # capped
    assert raddists["tube"]._grid_spacing() == pytest.approx(1e-3)
    monkeypatch.setitem(raddists["ring"].info, "gridSpacing", 0.002)
    assert raddists["ring"]._grid_spacing() == 0.002


def test_library_config_method_and_validation():
    base = {"tokamakName": "MOCK", "shot": 1, "injectionLocation": 225, "grid": {"NumR": 2, "NumZ": 2},
            "types": {"r": {"distType": "ElongatedRing", "sigma_R": 0.2, "sigma_z": 0.2}}}
    lib = RadDistLibrary.from_dict({**base, "rays": {"method": "python", "gridSpacing": 0.002}})
    cfg = lib.radDist_config(lib.types[0], lib.shape_sets(lib.types[0])[0])
    assert cfg["rayMethod"] == "python" and cfg["gridSpacing"] == 0.002
    lib = RadDistLibrary.from_dict(base)
    assert lib.radDist_config(lib.types[0], lib.shape_sets(lib.types[0])[0])["rayMethod"] == "compiled"
    with pytest.raises(ValueError, match="rays.method"):
        RadDistLibrary.from_dict({**base, "rays": {"method": "fast"}})


# ---------------------------------------------------------------------------
# Ray tracing: the camera signals agree, and compiled is much faster
# ---------------------------------------------------------------------------

def _observe(rd, rays, method, seed, cameras=None):
    rd.info["rayMethod"] = method
    rd.info["BOLOMETER_PROPS"] = {"pixelSamples": rays, "seed": seed}
    if cameras is not None:
        rd.tokamak.bolometers = [b for b in rd.tokamak.bolometers if b.name in cameras]
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        rd.bolos_observe()
    data = rd.data["Brightness"]
    sig = np.concatenate([[float(np.sum(v)) for v in data[n][c]] for n in rd.info["emissionNames"] for c in data[n]])
    return sig, time.perf_counter() - t0


@pytest.mark.cherab
@pytest.mark.parametrize("kind", ["ring", "nimrod"])
def test_camera_signals_agree_and_compiled_is_faster(raddists, kind):
    rd = raddists[kind]
    cams = ["POL_A", "POL_C", "SKEW"]
    py, t_py = _observe(rd, 60, "python", 1, cams)
    _, t_c = _observe(rd, 60, "compiled", 2)
    ref, _ = _observe(rd, 2000, "compiled", 3)
    lit = ref > 0.05 * ref.max()
    ratio = py[lit] / ref[lit]
    assert abs(ratio.mean() - 1.0) < 4 * ratio.std(ddof=1) / np.sqrt(ratio.size) + 0.01   # no bias beyond the noise
    assert t_py / t_c > 5
