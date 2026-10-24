"""
tests/test_parametric.py

Fitting a model's parameters directly through the geometry matrix
(emis3d.parametric), with no radDist library: on a synthetic matrix (fast)
and end to end through Emis3D on MOCK (ray tracing).
"""

import contextlib
import io

import h5py
import numpy as np
import pytest
import scipy.sparse as sp
import yaml

import emis3d.radDist as RD
import emis3d.util_emis3D as U
from emis3d import mock
from emis3d.geometry import GeometryMatrix, GridSpec
from emis3d.parametric import MODELS, ParametricFit
from emis3d.run_config import RunConfig
from emis3d.workspace import Workspace, set_workspace

BASE = {"tokamakName": "MOCK", "units": "Brightness", "injectionLocation": 225.0, "saveRunsDirectoryName": "p", "BOLOMETER_PROPS": {"pixelSamples": 10}}
TRUE = {"startR": 1.75, "startZ": 0.15, "sigma_R": 0.25, "sigma_z": 0.2, "rotationAngle": 30.0, "b": 0.3, "a": 3.0}


@pytest.fixture(scope="module")
def rings(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ws")
    set_workspace(Workspace.resolve(workspace=tmp, env={}, user_config=tmp / "none.yaml"))
    with contextlib.redirect_stdout(io.StringIO()):
        truth = RD.ElongatedRing(startR=TRUE["startR"], startZ=TRUE["startZ"], config={**BASE, "distType": "ElongatedRing", "eqFileName": None,
                                 "sigma_R": TRUE["sigma_R"], "sigma_z": TRUE["sigma_z"], "rotationAngle": TRUE["rotationAngle"]})
        template = RD.ElongatedRing(startR=1.9, startZ=0.0, config={**BASE, "distType": "ElongatedRing", "eqFileName": None,
                                    "sigma_R": 0.1, "sigma_z": 0.1, "rotationAngle": 0.0})
    yield tmp, truth, template
    set_workspace(None)


def synthetic_G(truth, n_channels=40, seed=0):
    """A dense-ish random matrix on MOCK's wall grid: well-posed, no ray tracing."""
    g = GridSpec.from_wall(truth.tokamak.wall, {"dR": 0.05, "dz": 0.05, "n_phi": 36})
    rng = np.random.default_rng(seed)
    rows = sp.random(n_channels, g.n_voxels, density=0.02, random_state=rng, format="csr")
    RR, ZZ = np.meshgrid(g.r_centers, g.z_centers, indexing="ij")
    inside_rz = truth.tokamak._inside_tokamak(np.column_stack([RR.ravel(), ZZ.ravel()])).reshape(g.n_r, g.n_z)
    inside = np.repeat(inside_rz[:, None, :], g.n_phi, axis=1).ravel()
    return GeometryMatrix(machine="MOCK", grid=g, rows=rows, cameras=["A"], channels=[f"A{i:02d}" for i in range(n_channels)],
                          camera_of_row=np.zeros(n_channels, int), etendues=np.full(n_channels, 1e-7), inside_wall=inside, samples=1)


def fabricate(G, truth, noise=0.01, seed=1):
    eps = G.emissivity_on_grid(truth, "elongatedRing")
    f = np.exp(-TRUE["b"] * U.find_dphi(G.grid.phi_centers[G.sector_of_voxel()], np.deg2rad(225.0), "elongatedRing") ** 2)
    signal = TRUE["a"] * G.signals(eps * f)
    sigma = 0.05 * signal.max() + 0.05 * signal
    rng = np.random.default_rng(seed)
    return {t: (float(s + noise * s * rng.normal()), float(e)) for t, s, e in zip(G.channels, signal, sigma, strict=True)}, signal


def test_recovers_the_ring_from_scratch(rings):
    _, truth, template = rings
    G = synthetic_G(truth)
    data, _ = fabricate(G, truth)
    fit = ParametricFit("ring", template, [G], data, "exponential", 225.0, bounds={"rotationAngle": (0, 90)})
    res = fit.search(seed=0, maxiter=40)
    assert res.values["startR"] == pytest.approx(TRUE["startR"], abs=0.03)
    assert res.values["startZ"] == pytest.approx(TRUE["startZ"], abs=0.03)
    assert res.values["b"] == pytest.approx(TRUE["b"], abs=0.05)
    assert res.values["sigma_R"] == pytest.approx(TRUE["sigma_R"], abs=0.05) and res.values["sigma_z"] == pytest.approx(TRUE["sigma_z"], abs=0.05)
    assert res.chi2 / res.n_data < 2.0 and len(res.stages) == 2 and all(res.stderr[k] is not None for k in res.values)
    assert "startR" in res.summary()


def test_closed_form_amplitude_and_fixed_parameters(rings):
    _, truth, template = rings
    G = synthetic_G(truth)
    data, _ = fabricate(G, truth, noise=0.0)
    fit = ParametricFit("ring", template, [G], data, "exponential", 225.0,
                        fixed={k: TRUE[k] for k in ("startR", "startZ", "sigma_R", "sigma_z", "rotationAngle")})
    assert fit.names == ["b"]
    chi2, a = fit.chi2(G, {"b": TRUE["b"]})
    assert a == pytest.approx(1.0, rel=1e-6) and chi2 < 1e-12                   # relative: the model's peak is the data's
    chi2_wrong, _ = fit.chi2(G, {"b": 1.0})
    assert chi2_wrong > chi2 + 1.0


def test_constant_scale_has_no_decay_parameter(rings):
    _, truth, template = rings
    G = synthetic_G(truth)
    data, _ = fabricate(G, truth)
    fit = ParametricFit("ring", template, [G], data, "constant", 225.0)
    assert "b" not in fit.names and np.all(fit._toroidal(G, 0.0) == 1.0)


def test_channels_missing_from_the_matrix_are_an_error(rings):
    _, truth, template = rings
    G = synthetic_G(truth)
    data, _ = fabricate(G, truth)
    data["NOPE"] = (1.0, 1.0)
    with pytest.raises(ValueError, match="NOPE"):
        ParametricFit("ring", template, [G], data, "exponential", 225.0)
    with pytest.raises(ValueError, match="unknown model"):
        ParametricFit("banana", template, [G], {}, "exponential", 225.0)


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------

RUN = {"BOLOMETERS": {"A": {"dataFileName": "a.h5"}}, "shot": 1, "scale_def": "exponential"}


def test_run_config_solver_and_parametric_block():
    cfg = RunConfig.from_dict({**RUN, "solver": "parametric", "parametric": {"models": ["ring", "tube"], "matrices": ["low", "medium"],
                                                                            "bounds": {"ring": {"sigma_R": [0.05, 2.0]}}}})
    assert cfg.parametric["models"] == ["ring", "tube"] and cfg.radDistDirectories_LocIndependent == []
    with pytest.raises(ValueError, match="solver: library needs"):
        RunConfig.from_dict(RUN)
    with pytest.raises(ValueError, match="solver must be"):
        RunConfig.from_dict({**RUN, "solver": "magic"})
    with pytest.raises(ValueError, match="needs a parametric block"):
        RunConfig.from_dict({**RUN, "solver": "parametric"})
    bad = [
        ({"models": "ring"}, "must be a list"),
        ({"model": "ring"}, "use models"),
        ({"models": ["banana"]}, "unknown model"),
        ({"models": ["ring"], "bounds": {"sigma_R": [0, 1]}}, "per model"),
        ({"models": ["ring"], "bounds": {"ring": {"psi_center": [0, 1]}}}, "does not have"),
        ({"models": ["ring"], "bounds": {"ring": {"sigma_R": [1, 0]}}}, "low < high"),
        ({"models": ["ring"], "bounds": {"ring": {"b": [0, 1]}}}, "does not have"),
        ({"models": ["shell"]}, "equilibrium"),
    ]
    for block, match in bad:
        with pytest.raises(ValueError, match=match):
            RunConfig.from_dict({**RUN, "solver": "parametric", "parametric": block})
    assert set(MODELS) == {"ring", "tube", "shell"}


def test_run_config_a_and_b_bounds():
    lib = {**RUN, "radDistDirectories_LocIndependent": ["x"]}
    cfg = RunConfig.from_dict({**lib, "bounds": {"a": [0.1, 5.0], "b": [0.0, 2.0]}})
    assert cfg.amplitude_bounds() == (0.1, 5.0) and cfg.decay_bounds() == (0.0, 2.0)
    assert RunConfig.from_dict(lib).decay_bounds() == (0.0, 10.0)
    assert RunConfig.from_dict({**lib, "scale_def": "linear"}).decay_bounds() == (0.01, 0.4)
    for bounds, match in (({"c": [0, 1]}, "takes a and b"), ({"a": [2, 1]}, "low < high"), ({"a": [-1, 1]}, "below 0"),
                          ({"sigma_R": [0, 1]}, "parametric.bounds")):
        with pytest.raises(ValueError, match=match):
            RunConfig.from_dict({**lib, "bounds": bounds})


def test_library_parameters_take_the_configured_bounds():
    from emis3d.radDistFitting import RadDistFitting

    rd = RadDistFitting()
    rd.info.update(injectionLocation=0.0, emissionNames=["e"])
    rd.fitSynthetic = {"e": {"observed_phi_loc": [[[0.0]]]}}
    rd.create_parameters(enable_dphi_scaling=True, scale_def="exponential", a_bounds=(0.2, 3.0), b_bounds=(0.0, 1.5))
    p = rd.fitSynthetic["params"]["params"]
    assert (p["a_0"].min, p["a_0"].max) == (0.2, 3.0) and (p["b_e_0"].min, p["b_e_0"].max) == (0.0, 1.5)
    rd.create_parameters(enable_dphi_scaling=True, scale_def="linear")
    assert (rd.fitSynthetic["params"]["params"]["b_e_0"].min, rd.fitSynthetic["params"]["params"]["b_e_0"].max) == (0.01, 0.4)


# ---------------------------------------------------------------------------
# Comparing models, and parameters at their bounds
# ---------------------------------------------------------------------------

def test_at_bounds_names_the_parameter_and_side():
    from emis3d.parametric import at_bounds

    flags = at_bounds({"sigma_R": 1.5, "b": 0.0, "a": 1.0}, {"sigma_R": (0.03, 1.5), "b": (0.0, 10.0), "a": (0.05, 20.0)})
    assert flags == ["sigma_R at its upper bound 1.5", "b at its lower bound 0"]


def test_a_tight_bound_is_flagged(rings, caplog):
    _, truth, template = rings
    G = synthetic_G(truth)
    data, _ = fabricate(G, truth)
    fit = ParametricFit("ring", template, [G], data, "exponential", 225.0, bounds={"sigma_R": (0.03, 0.1), "rotationAngle": (0, 90)})
    res = fit.search(seed=0, maxiter=20)
    assert "sigma_R at its upper bound 0.1" in res.at_bounds and "AT BOUNDS" in res.summary()
    assert "sigma_R at its upper bound 0.1" in caplog.text


def test_amplitude_bounds_hold(rings):
    _, truth, template = rings
    G = synthetic_G(truth)
    data, _ = fabricate(G, truth, noise=0.0)
    fixed = {k: TRUE[k] for k in ("startR", "startZ", "sigma_R", "sigma_z", "rotationAngle")}
    fit = ParametricFit("ring", template, [G], data, "exponential", 225.0, fixed=fixed, a_bounds=(0.05, 0.5))
    chi2, a = fit.chi2(G, {"b": TRUE["b"]})
    assert a == pytest.approx(0.5) and chi2 > 1.0                               # held at its bound, so no longer exact


def test_models_are_compared_by_aic(rings):
    from emis3d.parametric import comparison_table

    _, truth, template = rings
    G = synthetic_G(truth)
    data, _ = fabricate(G, truth)
    with contextlib.redirect_stdout(io.StringIO()):
        tube = RD.SquareTube(startR=1.9, startZ=0.0, config={**BASE, "distType": "SquareTube", "eqFileName": None, "dR": 0.2, "dz": 0.2})
    results = [ParametricFit("ring", template, [G], data, "exponential", 225.0, bounds={"rotationAngle": (0, 90)}).search(seed=0, maxiter=30),
               ParametricFit("tube", tube, [G], data, "exponential", 225.0).search(seed=0, maxiter=30)]
    table = comparison_table(results).splitlines()
    assert table[0].split()[:5] == ["model", "chi2/N", "AIC", "dAIC", "k"]
    assert table[1].startswith("ring") and " 0.0 " in table[1]                 # the true model ranks first, delta AIC 0
    assert any(line.startswith("tube") for line in table[2:])
    assert results[0].n_params == 7                                             # 5 shape + b + a


# ---------------------------------------------------------------------------
# End to end through Emis3D on MOCK
# ---------------------------------------------------------------------------

@pytest.mark.cherab
def test_parametric_fit_end_to_end(rings):
    from emis3d.emis3D import Emis3D
    from emis3d.geometry import build_geometry_matrix, matrix_path

    tmp, truth, _ = rings
    (tmp / "emis3d.yaml").write_text("version: 1\n")
    ws = Workspace.resolve(workspace=tmp, env={}, user_config=tmp / "none.yaml")
    G = build_geometry_matrix(truth.tokamak, resolution={"dR": 0.05, "dz": 0.05, "n_phi": 72}, samples=3000)
    G.save(matrix_path(ws, "MOCK", "low"))

    data, _ = fabricate(G, truth, noise=0.02)
    times = np.linspace(9.0, 11.0, 201)
    meas = tmp / "measurements" / "MOCK" / "1"
    meas.mkdir(parents=True, exist_ok=True)
    for ci, cam in enumerate(G.cameras):
        rows = np.flatnonzero(G.camera_of_row == ci)
        vals = np.array([data[G.channels[k]][0] for k in rows])
        with h5py.File(meas / f"{cam}.h5", "w") as fh:
            fh.create_dataset("channelOrder", data=[G.channels[k] for k in rows])
            fh.create_dataset("TIME", data=times)
            fh.create_dataset("DATA_CALIBRATED", data=np.repeat(vals[:, None], times.size, axis=1))
    run = tmp / "runs" / "par" / "run.yaml"
    run.parent.mkdir(parents=True, exist_ok=True)
    run.write_text(yaml.safe_dump({"tokamakName": "MOCK", "shot": 1, "tstart": 10.0, "dt": 0.05, "injectionLocation": 225.0,
                                   "scale_def": "exponential", "error_definition": "floor_relative",
                                   "solver": "parametric",
                                   "parametric": {"models": ["ring"], "matrices": ["low"], "maxiter": 40, "bounds": {"ring": {"rotationAngle": [0, 90]}}},
                                   "BOLOMETERS": {c: {"dataFileName": f"1/{c}.h5"} for c in G.cameras}}))
    e = Emis3D(runConfigName=str(run))
    assert e._perform_fits() == [10.0]
    best = e.bestFits[10.0]
    res = best["parametric"]
    assert res.values["startR"] == pytest.approx(TRUE["startR"], abs=0.05) and res.values["startZ"] == pytest.approx(TRUE["startZ"], abs=0.05)
    assert res.values["b"] == pytest.approx(TRUE["b"], abs=0.08)
    assert "ranges" in best and best["radiation_distribution"]["toroidal_peaking_factor"] > 1.0
    assert best["radDistInfo"]["parametric"]["model"] == "ring"
    e._plot_bestFit(save=True)
    assert list((tmp / "runs" / "par" / "results").glob("*/images/*.png"))
    assert mock.NAME == "MOCK"


def test_check_reports_missing_matrices(tmp_path):
    from emis3d.pipeline import check
    from test_calibration import write_workspace

    run = write_workspace(tmp_path, tstart=10.0, radDistDirectories_LocIndependent=[], solver="parametric",
                          parametric={"models": ["ring"], "matrices": ["nope"]})
    msgs = [f.message for f in check(run) if f.level == "error"]
    assert any("geometry matrix 'nope' not found" in m and "emis3d build-matrix MOCK --resolution nope" in m for m in msgs)


# ---------------------------------------------------------------------------
# Data channels that the geometry matrix does not have
# ---------------------------------------------------------------------------

def matrix_for(rings, drop=(), cameras=None):
    """A synthetic matrix with MOCK's real channel names, minus `drop`."""
    from test_calibration import CAMS, cam_info

    _, truth, _ = rings
    g = GridSpec.from_wall(truth.tokamak.wall, {"dR": 0.1, "dz": 0.1, "n_phi": 12})
    tags, cam_rows = [], []
    for ci, cam in enumerate(cameras or CAMS):
        for t in cam_info(cam)[0]:
            if t not in drop:
                tags.append(str(t))
                cam_rows.append(ci)
    rng = np.random.default_rng(0)
    rows = sp.random(len(tags), g.n_voxels, density=0.05, random_state=rng, format="csr")
    return GeometryMatrix(machine="MOCK", grid=g, rows=rows, cameras=list(cameras or CAMS), channels=tags,
                          camera_of_row=np.array(cam_rows), etendues=np.full(len(tags), 1e-7),
                          inside_wall=np.ones(g.n_voxels, bool), samples=1)


def test_channels_missing_from_the_matrix_are_dropped(rings, tmp_path, caplog):
    from emis3d.emis3D import Emis3D
    from emis3d.geometry import matrix_path
    from test_calibration import CAMS, cam_info, write_workspace

    missing = str(cam_info("POL_A")[0][3])
    bad = str(cam_info("POL_C")[0][2])
    run = write_workspace(tmp_path, tstart=10.0, radDistDirectories_LocIndependent=[], solver="parametric", scale_def="exponential",
                          error_definition="floor_relative", parametric={"models": ["ring"], "matrices": ["low"], "maxiter": 3},
                          BOLOMETERS={c: {"dataFileName": f"1/{c}.h5", **({"badChannels": [bad]} if c == "POL_C" else {})} for c in CAMS})
    ws = Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml")
    matrix_for(rings, drop={missing}).save(matrix_path(ws, "MOCK", "low"))

    e = Emis3D(runConfigName=str(run))
    e._fit_parametric(evalTime=10.0)                                     # used to raise: channel not in the matrix
    assert missing in caplog.text and "Dropped 1 channel" in caplog.text
    kept = [ch for chans in e.channel_order["channel_list"] for ch in chans]
    assert missing not in kept and bad in kept                           # a bad channel stays (excluded by its error)
    assert e.parametric[10.0].n_data == len(kept) - 1                    # ... but is not fitted


def test_a_camera_missing_from_the_matrix_is_an_error(rings, tmp_path):
    from emis3d.emis3D import Emis3D
    from emis3d.geometry import matrix_path
    from test_calibration import write_workspace

    run = write_workspace(tmp_path, tstart=10.0, radDistDirectories_LocIndependent=[], solver="parametric", scale_def="exponential",
                          parametric={"models": ["ring"], "matrices": ["low"], "maxiter": 3})
    ws = Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml")
    matrix_for(rings, cameras=["POL_A", "POL_B", "POL_C", "WRAP"]).save(matrix_path(ws, "MOCK", "low"))   # no SKEW
    e = Emis3D(runConfigName=str(run))
    with pytest.raises(ValueError, match="camera SKEW has no channels in the geometry matrices"):
        e._fit_parametric(evalTime=10.0)
