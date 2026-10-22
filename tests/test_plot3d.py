"""
tests/test_plot3d.py

Rebuilding saved radDists of every type (radDist.from_info / radDist.load),
the emission point cloud, and the 3-D view (RadDist.plot3D) on MOCK.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import emis3d.radDist as RD
from emis3d import mock
from emis3d.workspace import Workspace, set_workspace

BASE = {"tokamakName": "MOCK", "units": "Brightness", "injectionLocation": 225.0, "saveRunsDirectoryName": "p",
        "BOLOMETER_PROPS": {"pixelSamples": 10, "seed": 0}}
RING = {**BASE, "distType": "ElongatedRing", "eqFileName": None, "sigma_R": 0.15, "sigma_z": 0.2, "rotationAngle": 0.0}
HELICAL = {**BASE, "distType": "Helical", "eqFileName": mock.EQ_FILE, "sigma_R": 0.05, "sigmaKernel": 0.03,
           "numFieldLines": 20, "numTransists": 1.0}


@pytest.fixture(autouse=True)
def workspace(tmp_path):
    set_workspace(Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml"))
    yield tmp_path
    set_workspace(None)
    plt.close("all")


def simulation_file(path, axes=("z", "r", "phi")):
    """A blob with an n = 1 toroidal variation peaking at 225 deg."""
    r = np.linspace(1.05, 2.35, 27)
    z = np.linspace(-1.05, 1.05, 43)
    phi = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    grids = dict(zip(axes, np.meshgrid(*({"r": r, "z": z, "phi": phi}[a] for a in axes), indexing="ij"), strict=True))
    prad = np.exp(-((grids["r"] - 1.85) ** 2 / 0.08 + (grids["z"] - 0.1) ** 2 / 0.125)) * (1 + 0.8 * np.cos(grids["phi"] - np.deg2rad(225)))
    path.write_text(json.dumps({"pradArray": prad.tolist(), "r_list": r.tolist(), "z_list": z.tolist(), "phi_list": phi.tolist()}))
    return path


def emissivity(rd, R, Z, phi):
    return sum(rd.calc_emissivity(np.array([R]), np.array([Z]), phi=np.array([phi]), emissionName=n)[n][0]
               for n in rd.info["emissionNames"])


# ---------------------------------------------------------------------------
# Rebuilding radDists
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda d: RD.ElongatedRing(startR=1.85, startZ=0.1, config=dict(RING)),
    lambda d: RD.Helical(startR=1.95, startZ=0.1, config=dict(HELICAL)),
    lambda d: RD.NIMROD(nimrodFile=str(simulation_file(d / "n.json")), timestep=40, config={**BASE, "eqFileName": None}),
    lambda d: RD.M3DC1(m3dc1File=str(simulation_file(d / "m.json", ("phi", "r", "z"))), timestep=7, config={**BASE, "eqFileName": None}),
], ids=["ring", "helical", "nimrod", "m3dc1"])
def test_from_info_rebuilds_every_type(make, tmp_path):
    original = make(tmp_path)
    rebuilt = RD.from_info(original.info)
    assert type(rebuilt) is type(original)
    for R, Z, deg in ((1.85, 0.1, 225.0), (1.95, 0.1, 200.0), (1.6, -0.3, 45.0)):
        assert emissivity(rebuilt, R, Z, np.deg2rad(deg)) == pytest.approx(emissivity(original, R, Z, np.deg2rad(deg)), rel=1e-9)


def test_load_rebuilds_a_saved_simulation_raddist(tmp_path, workspace):
    nim = RD.NIMROD(nimrodFile=str(simulation_file(tmp_path / "shot1.json")), timestep=40, config={**BASE, "eqFileName": None})
    nim.data = {"Brightness": {}}
    nim.saveRadDist()
    saved = next((workspace / "raddists" / "MOCK" / "p").glob("**/*.json"))
    assert saved.name == "t_40.json"
    loaded = RD.load(saved)
    assert isinstance(loaded, RD.NIMROD) and loaded.info["timestep"] == 40
    assert emissivity(loaded, 1.85, 0.1, np.deg2rad(225)) == pytest.approx(emissivity(nim, 1.85, 0.1, np.deg2rad(225)))


def test_unknown_type_is_an_error():
    with pytest.raises(ValueError, match="Cannot rebuild"):
        RD.from_info({"distType": "Banana"})


# ---------------------------------------------------------------------------
# Point cloud
# ---------------------------------------------------------------------------

def test_cloud_is_inside_the_wall_and_centred_on_the_emission():
    ring = RD.ElongatedRing(startR=1.85, startZ=0.1, config=dict(RING))
    c = ring.emission_cloud(n_points=20000, n_phi=36, spacing=0.03)
    R = np.hypot(c["x"], c["y"])
    assert ring.tokamak._inside_tokamak(np.column_stack([R, c["z"]])).mean() > 0.99
    assert np.average(R) == pytest.approx(1.85, abs=0.05) and np.average(c["z"]) == pytest.approx(0.1, abs=0.05)


def test_cloud_density_follows_toroidal_variation(tmp_path):
    nim = RD.NIMROD(nimrodFile=str(simulation_file(tmp_path / "n.json")), timestep=1, config={**BASE, "eqFileName": None})
    c = nim.emission_cloud(n_points=20000, n_phi=36, spacing=0.03)
    phi = np.arctan2(c["y"], c["x"])
    near = np.mean(np.cos(phi - np.deg2rad(225)) > 0)          # half of the torus centred on 225 deg
    assert near == pytest.approx(0.5 + 0.8 / np.pi, abs=0.03)  # (1 + 0.8 cos) puts 75% of the emission there


def test_toroidal_scale_reweights_the_cloud():
    ring = RD.ElongatedRing(startR=1.85, startZ=0.1, config=dict(RING))
    c = ring.emission_cloud(n_points=20000, n_phi=36, spacing=0.03, toroidal_scale=lambda p: 1.0 if np.cos(p) > 0 else 0.0)
    assert np.all(c["x"] > -0.2)                                 # only the x > 0 half (plus jitter)


# ---------------------------------------------------------------------------
# The 3-D view
# ---------------------------------------------------------------------------

def test_chords_are_coloured_by_camera_height(tmp_path):
    cams = list(mock.DEFAULT.cameras) + [mock.CameraSpec("MID", 100.0, z=0.0, rotation_deg=90.0)]
    set_workspace(mock.mock_workspace(tmp_path, cameras=cams))
    ring = RD.ElongatedRing(startR=1.85, startZ=0.1, config=dict(RING))
    tok = ring.tokamak
    assert {b.name: tok.camera_height_class(b) for b in tok.bolometers} == {
        "POL_A": "above", "POL_B": "below", "POL_C": "above", "SKEW": "above", "WRAP": "above", "MID": "midplane"}
    ax = ring.plot3D(n_points=2000)
    import inspect

    chord_lw = inspect.signature(tok.plot3d_chords).parameters["lw"].default   # follows any restyling
    colours = {ln.get_color() for ln in ax.get_lines() if ln.get_linewidth() == chord_lw}
    assert colours == set(tok.HEIGHT_COLORS.values())
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "cameras above the midplane", "cameras at the midplane", "cameras below the midplane"]


def test_chords_stop_at_the_wall():
    ring = RD.ElongatedRing(startR=1.85, startZ=0.1, config=dict(RING))
    tok = ring.tokamak
    for bolo in tok.bolometers:
        for foil in bolo.bolometer_camera:
            origin, hit, _ = foil.trace_sightline()
            end = tok._chord_end_at_wall(origin, hit)
            R = float(np.hypot(end[0], end[1]))
            assert R <= tok.wall["maxr"] + 0.01 and abs(end[2]) <= tok.wall["maxz"] + 0.01


def test_wall_has_thick_outlines_and_no_midplane_rings():
    ring = RD.ElongatedRing(startR=1.85, startZ=0.1, config=dict(RING))
    ax = ring.plot3D(n_points=2000, cameras=[], show_injector=False, show_field_lines=False)
    walls = [ln for ln in ax.get_lines()]
    assert len(walls) == 4                                        # four poloidal outlines, no rings
    assert all(ln.get_linewidth() == 2.0 for ln in walls)
    assert all(np.ptp(np.asarray(ln.get_data_3d()[2])) > 1.0 for ln in walls)   # all span z (none flat at z = 0)


def test_helical_view_draws_field_lines_and_simulation_title(tmp_path):
    hel = RD.Helical(startR=1.95, startZ=0.1, config=dict(HELICAL))
    ax = hel.plot3D(n_points=2000, cameras=[])
    assert len([ln for ln in ax.get_lines() if ln.get_color() == "black"]) == 2   # one per direction
    nim = RD.NIMROD(nimrodFile=str(simulation_file(tmp_path / "shot7.json")), timestep=40, config={**BASE, "eqFileName": None})
    assert nim.plot3D(n_points=2000, cameras=[]).get_title() == "NIMROD radDist: shot7, timestep 40"
