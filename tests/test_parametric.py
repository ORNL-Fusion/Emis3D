"""
tests/test_parametric.py

Fitting a model's parameters directly through the geometry matrix
(emis3d.parametric), with no radDist library: on a synthetic matrix (fast)
and end to end through Emis3D on MOCK (ray tracing).
"""

import contextlib
import tempfile
from pathlib import Path
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
    assert set(MODELS) == {"ring", "tube", "shell", "asymshell", "helical", "helical_psi", "spray_spot"}


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


def test_tilt_at_zero_is_not_flagged(rings):
    _, truth, template = rings
    G = synthetic_G(truth)
    data, _ = fabricate(G, truth)
    fit = ParametricFit("ring", template, [G], data, "exponential", 225.0, fixed={k: TRUE[k] for k in ("startR", "startZ", "sigma_R", "sigma_z")},
                        bounds={"rotationAngle": (0.0, 90.0)})
    res = fit.search(seed=0, maxiter=10)
    assert not any("rotationAngle" in f for f in res.at_bounds)


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


# ---------------------------------------------------------------------------
# The step scale: one amplitude per camera location
# ---------------------------------------------------------------------------

STEP_TRUE = {20.0: 0.5, 45.0: 1.0, 165.0: 0.8, 285.0: 0.3}   # MOCK's camera angles (POL_B and SKEW share 165)


def step_data(fit, G, shape):
    X = fit.design(G, shape)
    A = np.array([STEP_TRUE[d] for d in fit.step_deg])
    signal = X @ A
    sigma = 0.05 * signal.max() + 0.05 * signal
    _, rows_d, _ = fit._rows[id(G)]
    tags = [G.channels[k] for k in fit._rows[id(G)][0]]
    return {t: (float(s), float(e)) for t, s, e in zip(tags, signal, sigma, strict=True)}


def test_step_solves_one_amplitude_per_camera_location(rings):
    _, truth, template = rings
    G = matrix_for(rings)
    shape = {k: TRUE[k] for k in ("startR", "startZ", "sigma_R", "sigma_z", "rotationAngle")}
    data0 = {t: (1.0, 1.0) for t in G.channels}
    probe = ParametricFit("ring", template, [G], data0, "step", 225.0, fixed=shape)
    assert probe.step_deg == [20.0, 45.0, 165.0, 285.0] and probe.amp_names == ["a_20", "a_45", "a_165", "a_285"]
    assert "b" not in probe.names
    data = step_data(probe, G, {})
    fit = ParametricFit("ring", template, [G], data, "step", 225.0, fixed=shape)
    chi2, a = fit.chi2(G, {})
    assert chi2 < 1e-10
    want = np.array([STEP_TRUE[d] for d in fit.step_deg])
    np.testing.assert_allclose(np.asarray(a) / np.max(a), want / want.max(), rtol=1e-6)      # the ratios, exactly


def test_step_search_reports_every_amplitude(rings):
    _, truth, template = rings
    G = matrix_for(rings)
    fixed = {k: TRUE[k] for k in ("startZ", "sigma_R", "sigma_z", "rotationAngle")}
    probe = ParametricFit("ring", template, [G], {t: (1.0, 1.0) for t in G.channels}, "step", 225.0, fixed={**fixed, "startR": TRUE["startR"]})
    data = step_data(probe, G, {})
    fit = ParametricFit("ring", template, [G], data, "step", 225.0, fixed=fixed, bounds={"startR": (1.6, 1.9)})
    res = fit.search(seed=0, maxiter=15)
    assert res.values["startR"] == pytest.approx(TRUE["startR"], abs=0.02)
    assert set(res.amplitude) == {"a_20", "a_45", "a_165", "a_285"} and res.n_params == 1 + 4
    assert res.toroidal["locations_deg"] == [20.0, 45.0, 165.0, 285.0]
    rel = np.array(res.toroidal["amplitudes"])
    want = np.array([STEP_TRUE[d] for d in res.toroidal["locations_deg"]])
    np.testing.assert_allclose(rel / rel.max(), want / want.max(), rtol=0.02)


def test_fitted_step_is_built_into_the_candidate(rings):
    from emis3d.emis3D import Emis3D

    _, truth, template = rings
    G = matrix_for(rings)
    template.info.update({k: TRUE[k] for k in ("startR", "startZ", "sigma_R", "sigma_z", "rotationAngle")})
    template.calc_radiated_power()
    template.observe_with_matrix(G)
    before = {cam: [np.array(v, float) for v in segs] for cam, segs in template.data["Brightness"]["elongatedRing"].items()}
    Emis3D._apply_fitted_step(template, {"locations_deg": [45.0, 225.0], "amplitudes": [2.0, 1.0]})
    prof = template.data["toroidalRadiatedPower"]["elongatedRing"]
    phi = np.rad2deg(np.asarray(prof["phi_array"]))
    P = np.asarray(prof["P_pol"])
    near_45 = np.abs(((phi - 45 + 180) % 360) - 180) < 90
    assert np.allclose(P[~near_45] / P[near_45].mean(), 0.5, rtol=1e-6)                       # two levels, ratio 1 : 0.5
    phis = template.data["observed_phi_loc"]["elongatedRing"]
    for cam, segs in template.data["Brightness"]["elongatedRing"].items():
        for v, v0, p in zip(segs, before[cam], phis[cam], strict=True):
            d = np.rad2deg(np.asarray(p))
            f = np.where(np.abs(((d - 45 + 180) % 360) - 180) <= 90 + 1e-9, 1.0, 0.5)   # a tie goes to the first location
            np.testing.assert_allclose(v, v0 * f, rtol=1e-12)


def test_step_end_to_end_through_emis3d(rings, tmp_path):
    from emis3d.emis3D import Emis3D
    from emis3d.geometry import matrix_path
    from test_calibration import write_workspace

    run = write_workspace(tmp_path, tstart=10.0, radDistDirectories_LocIndependent=[], solver="parametric", scale_def="step",
                          error_definition="floor_relative",
                          parametric={"models": ["ring"], "matrices": ["low"], "maxiter": 5,
                                      "bounds": {"ring": {"startR": [1.6, 1.9], "startZ": [0.0, 0.3]}}})
    ws = Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml")
    matrix_for(rings).save(matrix_path(ws, "MOCK", "low"))
    e = Emis3D(runConfigName=str(run))
    assert e.info["scale_def"] == "constant" and e.config.scale_def == "step"
    e._perform_fits()
    best = e.bestFits[10.0]
    res = best["parametric"]
    assert res.toroidal["locations_deg"] == [20.0, 45.0, 165.0, 285.0]
    P = np.asarray(best["radiation_distribution"]["total_power"])
    assert len(np.unique(np.round(P / P.max(), 9))) <= 4                                        # a step profile
    assert best["radiation_distribution"]["toroidal_peaking_factor"] == pytest.approx(P.max() / P.mean())
    assert "scale_def: step" in e._figure_title(10.0)
    lo, hi = best["ranges"]["total"]["tpf"]["68%"]
    tpf = best["radiation_distribution"]["toroidal_peaking_factor"]
    assert hi > lo and lo - 1e-9 <= tpf <= hi + 1e-9                                         # a real range, containing the best fit
    assert "covariance" in best["ranges"]["note"]


# ---------------------------------------------------------------------------
# The asymmetric shell and the helical model
# ---------------------------------------------------------------------------

def test_asymmetric_shell_is_brighter_on_its_side(rings):
    with contextlib.redirect_stdout(io.StringIO()):
        shell = RD.FluxShell(config={**BASE, "distType": "FluxShell", "eqFileName": mock.EQ_FILE, "psi_center": 0.8, "psi_width": 0.1,
                                     "asym_amp": 1.0, "asym_angle": 90.0})                   # bright at the top
    R0, z0 = shell.info["startR"], shell.info["startZ"]
    top = shell.calc_emissivity(np.array([R0]), np.array([z0 + 0.55]), phi=np.array([0.0]), emissionName="fluxShell")["fluxShell"][0]
    bottom = shell.calc_emissivity(np.array([R0]), np.array([z0 - 0.55]), phi=np.array([0.0]), emissionName="fluxShell")["fluxShell"][0]
    assert top > 0 and bottom == 0.0                                                       # 1 + cos(180 deg) = 0 on the far side
    shell.info["asym_amp"] = 0.0
    sym = shell.calc_emissivity(np.array([R0, R0]), np.array([z0 + 0.55, z0 - 0.55]), phi=np.array([0.0]), emissionName="fluxShell")["fluxShell"]
    assert sym[0] == pytest.approx(top / 2) and sym[1] > 0
    assert set(MODELS) == {"ring", "tube", "shell", "asymshell", "helical", "helical_psi", "spray_spot"}


@pytest.fixture(scope="module")
def helical_setup(rings, tmp_path_factory):
    """A synthetic matrix with MOCK's cameras, its field-line mapping, and a field-aligned template."""
    from emis3d.fieldline_map import FieldLineMap
    from emis3d.tokamak import Tokamak

    tmp, truth, _ = rings
    G = matrix_for(rings)                                                                    # 10 cm, 12 sectors: fast to trace
    ws = Workspace.resolve(workspace=tmp, env={}, user_config=tmp / "none.yaml")
    gfile = Tokamak(tokamakName="MOCK", mode="Analysis", eqFileName=mock.EQ_FILE, workspace=ws).gfile
    fmap = FieldLineMap.build(G, gfile, np.deg2rad(225.0), "test", mock.EQ_FILE, transits=1, substeps=1)
    path = fmap.save(tmp_path_factory.mktemp("map") / "map.npz")
    with contextlib.redirect_stdout(io.StringIO()):
        template = RD.FieldAlignedSource(config={**BASE, "distType": "FieldAligned", "eqFileName": mock.EQ_FILE, "mapPath": str(path),
                                                 "startR": 1.9, "startZ": 0.0, "sigma_R": 0.1, "sigma_z": 0.1, "decay": 0.0})
    return G, fmap, template


HELICAL_TRUE = {"startR": 1.95, "startZ": 0.10, "sigma_R": 0.12, "sigma_z": 0.12, "decay": 0.3}


def test_helical_model_recovers_a_field_aligned_source(helical_setup):
    G, fmap, template = helical_setup
    probe = ParametricFit("helical", template, [G], {t: (1.0, 1.0) for t in G.channels}, "constant", 225.0)
    assert probe.names == ["startR", "startZ", "sigma_R", "sigma_z", "decay"] and probe.field_aligned and not probe.has_decay
    signal = probe.model_signals(G, HELICAL_TRUE)
    sigma = 0.05 * signal.max() + 0.05 * signal
    data = {t: (float(s), float(e)) for t, s, e in zip(G.channels, signal, sigma, strict=True)}
    fit = ParametricFit("helical", template, [G], data, "constant", 225.0, bounds={"startR": (1.5, 2.2), "startZ": (-0.5, 0.5)})
    res = fit.search(seed=0, maxiter=40)
    assert res.values["startR"] == pytest.approx(HELICAL_TRUE["startR"], abs=0.06)
    assert res.values["startZ"] == pytest.approx(HELICAL_TRUE["startZ"], abs=0.06)
    assert res.values["decay"] == pytest.approx(HELICAL_TRUE["decay"], abs=0.15)
    assert res.chi2 / res.n_data < 0.5
    # the radDist itself: its emissivity off-grid interpolates the voxel field, and it observes through G
    template.info.update(HELICAL_TRUE)
    eps = template.voxel_emissivity(G)
    assert eps.max() > 0 and (eps >= 0).all()
    on = template.calc_emissivity(np.array([1.95]), np.array([0.10]), phi=np.array([np.deg2rad(225.0)]), emissionName="fieldAligned")["fieldAligned"][0]
    assert on > 0.3 * eps.max()
    sig = G.observe(template)["fieldAligned"]
    np.testing.assert_allclose(np.concatenate([sig[c] for c in G.cameras]), G.signals(G.emissivity_on_grid(template, "fieldAligned")), rtol=1e-9)


def test_helical_and_ring_compared_end_to_end(rings, tmp_path, monkeypatch):
    from emis3d.emis3D import Emis3D
    from emis3d.fieldline_map import map_path
    from emis3d.geometry import matrix_path
    from test_calibration import write_workspace

    run = write_workspace(tmp_path, tstart=10.0, radDistDirectories_LocIndependent=[], solver="parametric", scale_def="constant",
                          parametric={"models": ["helical", "ring"], "matrices": ["low"], "equilibrium": mock.EQ_FILE, "maxiter": 4,
                                      "substeps": 1, "bounds": {"helical": {"startR": [1.6, 2.1], "startZ": [-0.3, 0.3]},
                                                                "ring": {"startR": [1.6, 2.1], "startZ": [-0.3, 0.3]}}})
    ws = Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml")
    matrix_for(rings).save(matrix_path(ws, "MOCK", "low"))
    e = Emis3D(runConfigName=str(run))
    e._perform_fits()
    best = e.bestFits[10.0]
    assert {r.model for r in best["parametric_models"]} == {"helical", "ring"}
    assert map_path(ws, "MOCK", "low", mock.EQ_FILE, float(e.info["injectionLocation"]), 1).is_file()   # built once, cached
    helical = next(r for r in best["parametric_models"] if r.model == "helical")
    assert "decay" in helical.values and helical.n_params == 6


# ---------------------------------------------------------------------------
# The helical source plane's toroidal angle (vary_peak_rad_location)
# ---------------------------------------------------------------------------

def test_rotated_source_plane_is_recovered(helical_setup):
    G, fmap, template = helical_setup
    truth = {**HELICAL_TRUE, "phi_source": 225.0 + 40.0}                                    # 40 deg past the injector
    probe = ParametricFit("helical", template, [G], {t: (1.0, 1.0) for t in G.channels}, "constant", 225.0, vary_peak=True,
                          peak_shift_max_degrees=90.0)
    assert "phi_source" in probe.names and probe.bounds["phi_source"] == (135.0, 315.0)
    signal = probe.model_signals(G, truth)
    sigma = 0.05 * signal.max() + 0.05 * signal
    data = {t: (float(s), float(e)) for t, s, e in zip(G.channels, signal, sigma, strict=True)}
    fit = ParametricFit("helical", template, [G], data, "constant", 225.0, vary_peak=True, peak_shift_max_degrees=90.0,
                        bounds={"startR": (1.5, 2.2), "startZ": (-0.5, 0.5)})
    res = fit.search(seed=0, maxiter=60)
    assert res.values["phi_source"] == pytest.approx(265.0, abs=12.0)
    assert res.values["startR"] == pytest.approx(HELICAL_TRUE["startR"], abs=0.08)
    assert res.chi2 / res.n_data < 0.5

    fixed = ParametricFit("helical", template, [G], data, "constant", 225.0)                 # not varied: at the injector
    assert "phi_source" not in fixed.names and fixed.fixed["phi_source"] == 225.0
    assert fixed.chi2(G, {**HELICAL_TRUE})[0] > res.chi2 + 1.0                                # which cannot explain rotated data


# ---------------------------------------------------------------------------
# helical_psi: a source in flux coordinates
# ---------------------------------------------------------------------------

PSI_TRUE = {"psi_center": 0.8, "psi_width": 0.12, "theta_center": 120.0, "theta_width": 40.0, "decay": 0.5}   # widths the 10 cm test grid resolves


@pytest.fixture(scope="module")
def psi_template(helical_setup):
    _, _, rz_template = helical_setup
    with contextlib.redirect_stdout(io.StringIO()):
        return RD.FieldAlignedSource(config={**BASE, "distType": "FieldAligned", "eqFileName": mock.EQ_FILE, "source": "psi",
                                             "mapPath": rz_template.info["mapPath"], **PSI_TRUE})


def test_psi_source_lies_on_its_flux_surface(psi_template):
    t = psi_template
    f = t.fmap
    src = t.source_on(f)
    psin, theta = t._flux_coordinates(f)
    on = src > 0.5
    assert on.sum() > 3 and np.all(np.abs(psin[on] - 0.8) < 0.15)                            # thin across surfaces
    assert np.rad2deg(np.ptp(theta[on])) > 40                                               # extended along one
    deg = np.rad2deg(theta[on])
    assert np.all((deg > 60) & (deg < 180))                                                 # around 120 deg (top-inboard)
    # the poloidal angle wraps: a band centred at 175 deg weighs +165 and -175 deg equally (no seam at +-180)
    t.info["theta_center"], t.info["theta_width"] = 175.0, 20.0
    t._flux_cache = {id(f): (np.full(3, 0.8), np.deg2rad([165.0, -175.0, 90.0]))}
    w = t.source_on(f)
    assert w[0] == pytest.approx(w[1], rel=1e-9) and w[2] < 1e-3
    t._flux_cache = {}
    t.info.update(PSI_TRUE)
    assert t.info["startR"] == pytest.approx(1.772, abs=0.01)                               # the magnetic axis


def test_helical_psi_recovers_a_band(helical_setup, psi_template):
    G, fmap, _ = helical_setup
    t = psi_template
    probe = ParametricFit("helical_psi", t, [G], {c: (1.0, 1.0) for c in G.channels}, "constant", 225.0)
    assert probe.names == ["psi_center", "psi_width", "theta_center", "theta_width", "decay"]
    signal = probe.model_signals(G, PSI_TRUE)
    sigma = 0.05 * signal.max() + 0.05 * signal
    data = {c: (float(s), float(e)) for c, s, e in zip(G.channels, signal, sigma, strict=True)}
    fit = ParametricFit("helical_psi", t, [G], data, "constant", 225.0, bounds={"psi_width": (0.05, 0.3), "theta_width": (5.0, 90.0)})
    res = fit.search(seed=0, maxiter=60)
    assert res.values["psi_center"] == pytest.approx(0.8, abs=0.1)
    assert abs(((res.values["theta_center"] - 120.0 + 180) % 360) - 180) < 25
    assert res.chi2 / res.n_data < 0.5
    assert not any("theta_center" in fl for fl in res.at_bounds)
    assert "psi_0.8" in t._save_names()[0] and "theta_120" in t._save_names()[0]


def test_helical_psi_through_emis3d(rings, tmp_path):
    from emis3d.emis3D import Emis3D
    from emis3d.geometry import matrix_path
    from test_calibration import write_workspace

    run = write_workspace(tmp_path, tstart=10.0, radDistDirectories_LocIndependent=[], solver="parametric", scale_def="constant",
                          parametric={"models": ["helical_psi"], "matrices": ["low"], "equilibrium": mock.EQ_FILE, "maxiter": 4, "substeps": 1})
    ws = Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml")
    matrix_for(rings).save(matrix_path(ws, "MOCK", "low"))
    e = Emis3D(runConfigName=str(run))
    e._perform_fits()
    res = e.bestFits[10.0]["parametric"]
    assert res.model == "helical_psi" and {"psi_center", "theta_width", "decay"} <= set(res.values)
    assert "psi" in e._raddist_description(10.0) and "theta" in e._raddist_description(10.0)


# ---------------------------------------------------------------------------
# Composites: two structures fitted together
# ---------------------------------------------------------------------------

def composite_components(helical_setup, psi_template):
    _, _, spot = helical_setup
    return [("band", "helical_psi", psi_template, {"psi_width": (0.05, 0.3), "theta_width": (5.0, 90.0)}, None),
            ("spot", "helical", spot, {"startR": (1.6, 2.2), "startZ": (-0.4, 0.4), "sigma_R": (0.05, 0.2), "sigma_z": (0.05, 0.2)}, None)]


COMPOSITE_TRUE = {"band_psi_center": 0.8, "band_psi_width": 0.12, "band_theta_center": 120.0, "band_theta_width": 40.0,
                  "band_decay": 0.5, "band_phi_source": 200.0, "spot_startR": 1.95, "spot_startZ": 0.1, "spot_sigma_R": 0.1,
                  "spot_sigma_z": 0.1, "spot_decay": 2.0, "spot_phi_source": 240.0}


def test_composite_names_amplitudes_and_consistency(helical_setup, psi_template):
    from emis3d.parametric import CompositeFit

    G, _, _ = helical_setup
    comps = composite_components(helical_setup, psi_template)
    fit = CompositeFit("band_spot", comps, [G], {c: (1.0, 1.0) for c in G.channels}, "constant", 225.0, vary_peak=True, peak_shift_max_degrees=90.0)
    assert fit.amp_names == ["a_band", "a_spot"] and fit.a_bounds[0] == 0.0
    assert {"band_decay", "band_phi_source", "spot_decay", "spot_phi_source"} <= set(fit.names)
    cols = fit.voxel_columns(G, COMPOSITE_TRUE)
    rows = fit._rows[id(G)][0]
    np.testing.assert_allclose(np.column_stack([G.signals(c)[rows] for c in cols]), fit.design(G, COMPOSITE_TRUE), rtol=1e-12)

    shared = CompositeFit("band_spot", comps, [G], {c: (1.0, 1.0) for c in G.channels}, "constant", 225.0, shared_toroidal=True,
                          vary_peak=True, peak_shift_max_degrees=90.0)
    assert "decay" in shared.names and "phi_peak" in shared.names and not any(n.endswith("_phi_source") for n in shared.names)
    fixed_peak = CompositeFit("band_spot", comps, [G], {c: (1.0, 1.0) for c in G.channels}, "constant", 225.0)
    assert fixed_peak.fixed["band_phi_source"] == 225.0 and "band_phi_source" not in fixed_peak.names


def test_composite_recovers_two_structures_at_different_toroidal_locations(helical_setup, psi_template):
    from emis3d.parametric import CompositeFit

    G, _, _ = helical_setup
    comps = composite_components(helical_setup, psi_template)
    probe = CompositeFit("band_spot", comps, [G], {c: (1.0, 1.0) for c in G.channels}, "constant", 225.0, vary_peak=True, peak_shift_max_degrees=90.0)
    signal = probe.design(G, COMPOSITE_TRUE) @ np.array([1.0, 0.6])
    sigma = 0.05 * signal.max() + 0.05 * signal
    data = {c: (float(s), float(e)) for c, s, e in zip(G.channels, signal, sigma, strict=True)}
    fit = CompositeFit("band_spot", comps, [G], data, "constant", 225.0, vary_peak=True, peak_shift_max_degrees=90.0)
    chi2, a = fit.chi2(G, COMPOSITE_TRUE)
    assert chi2 < 1e-12 and a[1] / a[0] == pytest.approx(0.6, rel=1e-6)                   # amplitudes exact for the true shapes
    res = fit.search(seed=0, maxiter=40)
    for k in ("band_psi_center", "band_phi_source", "spot_startR", "spot_phi_source"):
        assert res.values[k] == pytest.approx(COMPOSITE_TRUE[k], abs=0.1 * max(1.0, abs(COMPOSITE_TRUE[k]) / 10))
    assert res.n_params == len(fit.names) + 2 and [c["name"] for c in res.components] == ["band", "spot"]
    assert res.components[1]["values"]["phi"] == pytest.approx(240.0, abs=5.0)


def test_composite_radDist_reproduces_the_fit(helical_setup, psi_template):
    from emis3d.parametric import CompositeFit

    G, _, _ = helical_setup
    comps = composite_components(helical_setup, psi_template)
    probe = CompositeFit("band_spot", comps, [G], {c: (1.0, 1.0) for c in G.channels}, "constant", 225.0, vary_peak=True, peak_shift_max_degrees=90.0)
    signal = probe.design(G, COMPOSITE_TRUE) @ np.array([1.0, 0.6])
    data = {c: (float(s), 0.05 * signal.max()) for c, s in zip(G.channels, signal, strict=True)}
    fit = CompositeFit("band_spot", comps, [G], data, "constant", 225.0, vary_peak=True, peak_shift_max_degrees=90.0)
    _, a = fit.chi2(G, COMPOSITE_TRUE)
    values = {**COMPOSITE_TRUE, "a_band": a[0], "a_spot": a[1]}
    with contextlib.redirect_stdout(io.StringIO()):
        rd = RD.Composite(fit, values, config={**BASE, "eqFileName": mock.EQ_FILE, "numBins": 36})
    np.testing.assert_allclose(G.signals(rd.voxel_emissivity(G)), signal, rtol=1e-9)        # the composite reproduces the data
    assert "band (helical_psi)" in rd.info["composite_description"] and "spot (helical)" in rd.info["composite_description"]
    rd.calc_radiated_power()
    P = np.asarray(rd.data["toroidalRadiatedPower"]["composite"]["P_pol"])
    assert P.max() > 0 and P.max() / P.mean() > 1.2                                         # toroidally peaked


def test_composite_through_emis3d(rings, tmp_path):
    from emis3d.emis3D import Emis3D
    from emis3d.geometry import matrix_path
    from test_calibration import write_workspace

    composites = {"band_spot": {"components": [{"model": "helical_psi", "name": "band"},
                                                {"model": "ring", "name": "core", "bounds": {"startR": [1.6, 1.9], "startZ": [0.0, 0.3]}}],
                                 "shared_toroidal": False}}
    run = write_workspace(tmp_path, tstart=10.0, radDistDirectories_LocIndependent=[], solver="parametric", scale_def="exponential",
                          parametric={"models": ["band_spot"], "matrices": ["low"], "equilibrium": mock.EQ_FILE, "maxiter": 3, "substeps": 1,
                                      "composites": composites})
    ws = Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml")
    matrix_for(rings).save(matrix_path(ws, "MOCK", "low"))
    e = Emis3D(runConfigName=str(run))
    e._perform_fits()
    best = e.bestFits[10.0]
    res = best["parametric"]
    assert res.model == "band_spot" and {"a_band", "a_core", "band_decay", "core_b"} <= set(res.values)
    assert e.info["scale_def"] == "exponential"                                             # the override is per time and restored
    assert "band (helical_psi)" in e._raddist_description(10.0)
    lo, hi = best["ranges"]["total"]["tpf"]["68%"]
    assert lo - 1e-9 <= best["radiation_distribution"]["toroidal_peaking_factor"] <= hi + 1e-9
    e._plot_bestFit(save=True)
    assert list((tmp_path / "runs").rglob("*.png"))


def test_composite_config_validation():
    base = {"BOLOMETERS": {"A": {"dataFileName": "a.h5"}}, "shot": 1, "solver": "parametric"}
    good = {"components": [{"model": "helical_psi", "name": "band"}, {"model": "helical", "name": "spot"}]}
    RunConfig.from_dict({**base, "parametric": {"models": ["bs"], "equilibrium": "g", "composites": {"bs": good}}})
    with pytest.raises(ValueError, match="equilibrium"):
        RunConfig.from_dict({**base, "parametric": {"models": ["bs"], "composites": {"bs": good}}})
    with pytest.raises(ValueError, match="unknown model"):
        RunConfig.from_dict({**base, "parametric": {"models": ["nope"], "composites": {"bs": good}}})
    with pytest.raises(ValueError, match="is a model's"):
        RunConfig.from_dict({**base, "parametric": {"models": ["ring"], "composites": {"ring": good}}})
    with pytest.raises(ValueError, match="shared_toroidal"):
        RunConfig.from_dict({**base, "parametric": {"models": ["bs"], "equilibrium": "g", "composites": {"bs": {**good, "shared_toroidal": "yes"}}}})


def test_composite_starts_from_its_components_fitted_alone(helical_setup, psi_template):
    """Seeded with a component's own best fit, the composite can only do at least as well as that component."""
    from emis3d.parametric import CompositeFit

    G, _, spot = helical_setup
    probe = ParametricFit("helical", spot, [G], {c: (1.0, 1.0) for c in G.channels}, "constant", 225.0)
    signal = probe.model_signals(G, HELICAL_TRUE)
    sigma = 0.05 * signal.max() + 0.05 * signal
    data = {c: (float(s), float(e)) for c, s, e in zip(G.channels, signal, sigma, strict=True)}
    alone = ParametricFit("helical", spot, [G], data, "constant", 225.0, bounds={"startR": (1.5, 2.2), "startZ": (-0.5, 0.5)}).search(seed=0, maxiter=30)
    comps = composite_components(helical_setup, psi_template)
    comps[1] = ("spot", "helical", spot, {"startR": (1.5, 2.2), "startZ": (-0.5, 0.5)}, None)
    fit = CompositeFit("band_spot", comps, [G], data, "constant", 225.0)
    start = fit.start_from({"helical": alone})
    assert start["spot_startR"] == alone.values["startR"] and start["spot_decay"] == alone.values["decay"]
    res = fit.search(seed=1, maxiter=3, start=start)                                          # a tiny budget
    assert res.chi2 <= alone.chi2 * 1.05 + 1e-9


def test_power_profile_counts_every_voxel_not_only_the_seen_ones(rings):
    """Radiated power must come from every voxel; evaluating only the voxels a camera sees made the profile
    peak where the cameras look (a TPF of 8 for a ring whose true TPF is 2)."""
    _, truth, template = rings
    G = synthetic_G(truth)                                                     # sparse rows: many voxels are unseen
    data, _ = fabricate(G, truth)
    fit = ParametricFit("ring", template, [G], data, "exponential", 225.0)
    values = {**{k: TRUE[k] for k in ("startR", "startZ", "sigma_R", "sigma_z", "rotationAngle")}, "b": 0.3, "a": 1.0}
    P = fit.sector_profile(G, values)
    envelope = np.exp(-0.3 * U.find_dphi(G.grid.phi_centers, np.deg2rad(225.0), "elongatedRing") ** 2)
    assert P.max() / P.mean() == pytest.approx(envelope.max() / envelope.mean(), rel=1e-6)   # a ring: the envelope's TPF
    seen_only = fit.voxel_field(G, values, only_seen=True)
    assert np.count_nonzero(seen_only) < np.count_nonzero(fit.voxel_field(G, values, only_seen=False))
    rows = fit._rows[id(G)][0]
    np.testing.assert_allclose(G.signals(seen_only)[rows], G.signals(fit.voxel_field(G, values, only_seen=False))[rows], rtol=1e-12)


def test_composite_power_includes_unseen_voxels(helical_setup, psi_template):
    from emis3d.parametric import CompositeFit

    G, _, _ = helical_setup
    comps = composite_components(helical_setup, psi_template)
    fit = CompositeFit("band_spot", comps, [G], {c: (1.0, 1.0) for c in G.channels}, "constant", 225.0, vary_peak=True, peak_shift_max_degrees=90.0)
    values = {**COMPOSITE_TRUE, "a_band": 1.0, "a_spot": 0.6}
    with contextlib.redirect_stdout(io.StringIO()):
        rd = RD.Composite(fit, values, config={**BASE, "eqFileName": mock.EQ_FILE, "numBins": 36})
    np.testing.assert_allclose(rd.voxel_emissivity(G), fit.voxel_field(G, values, only_seen=False))


def test_a_start_on_or_beyond_a_bound_is_accepted(rings):
    _, truth, template = rings
    G = synthetic_G(truth)
    data, _ = fabricate(G, truth)
    fit = ParametricFit("ring", template, [G], data, "exponential", 225.0, bounds={"rotationAngle": (0, 90)})
    hi = {n: fit.bounds[n][1] for n in fit.names}                                            # every value on its upper bound
    res = fit.search(seed=0, maxiter=2, start={**hi, "startR": float("nan"), "sigma_R": 99.0})
    assert np.isfinite(res.chi2)


# ---------------------------------------------------------------------------
# spray_spot: a spot on the SPI spray, in the injector's plane
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spray_template(helical_setup):
    import dataclasses

    from emis3d.injector import machine_injectors

    _, _, rz = helical_setup
    ws = Workspace.resolve(env={}, user_config=Path(tempfile.mktemp()))
    inj = machine_injectors(ws, "MOCK")["SPI225"]                                              # aimed at the axis, 20 deg cone
    with contextlib.redirect_stdout(io.StringIO()):
        t = RD.FieldAlignedSource(config={**BASE, "distType": "FieldAligned", "eqFileName": mock.EQ_FILE, "source": "spray",
                                          "injector": dataclasses.asdict(inj), "mapPath": rz.info["mapPath"],
                                          "sigma_R": 0.08, "sigma_z": 0.08, "decay": 0.5})
    return t, inj


SPRAY_TRUE = {"spray_s": 0.35, "spray_offset_deg": 4.0, "sigma_R": 0.08, "sigma_z": 0.08, "decay": 0.5}


def test_spray_source_lies_on_the_spray(spray_template):
    t, inj = spray_template
    g = t.tokamak.gfile
    assert 0.3 < t.info["spray_length"] < 2.0                                                  # to where the centre line leaves the vessel
    t.info.update(spray_s=0.0, spray_offset_deg=0.0)
    assert t.spray_position() == pytest.approx((inj.shatter_R, inj.shatter_Z))
    t.info.update(spray_s=0.3, spray_offset_deg=0.0)
    R0, Z0 = t.spray_position()
    assert inj.in_cone(R0, Z0, g.rmagx, g.zmagx) and np.hypot(R0 - inj.shatter_R, Z0 - inj.shatter_Z) == pytest.approx(0.3)
    t.info.update(spray_offset_deg=12.0)                                                       # past the 10 deg half-cone
    assert not inj.in_cone(*t.spray_position(), g.rmagx, g.zmagx)
    t.info.update(SPRAY_TRUE)
    src = t.source_on(t.fmap)
    peak = int(np.argmax(src))
    R0, Z0 = t.spray_position()
    assert abs(t.fmap.pixel_R[peak] - R0) < 0.1 and abs(t.fmap.pixel_z[peak] - Z0) < 0.1


def test_spray_spot_keeps_the_injector_plane_and_recovers_its_place(helical_setup, spray_template):
    G, _, _ = helical_setup
    t, inj = spray_template
    probe = ParametricFit("spray_spot", t, [G], {c: (1.0, 1.0) for c in G.channels}, "constant", 225.0, vary_peak=True)
    assert "phi_source" not in probe.names and probe.fixed["phi_source"] == 225.0                 # fragments fly in the injector's plane
    assert probe.bounds["spray_s"] == (0.0, t.info["spray_length"]) and probe.bounds["spray_offset_deg"] == (-10.0, 10.0)
    signal = probe.model_signals(G, SPRAY_TRUE)
    sigma = 0.05 * signal.max() + 0.05 * signal
    data = {c: (float(s), float(e)) for c, s, e in zip(G.channels, signal, sigma, strict=True)}
    res = ParametricFit("spray_spot", t, [G], data, "constant", 225.0, vary_peak=True).search(seed=0, maxiter=40)
    assert res.values["spray_s"] == pytest.approx(0.35, abs=0.06)
    assert res.values["spray_offset_deg"] == pytest.approx(4.0, abs=3.0)
    assert res.chi2 / res.n_data < 0.5


def test_band_with_a_spray_spot(helical_setup, psi_template, spray_template):
    from emis3d.parametric import CompositeFit

    G, _, _ = helical_setup
    t, _ = spray_template
    comps = [("band", "helical_psi", psi_template, {"psi_width": (0.05, 0.3), "theta_width": (5.0, 90.0)}, None),
             ("spot", "spray_spot", t, None, None)]
    fit = CompositeFit("band_spray", comps, [G], {c: (1.0, 1.0) for c in G.channels}, "constant", 225.0, vary_peak=True,
                       peak_shift_max_degrees=90.0)
    assert "band_phi_source" in fit.names and "spot_decay" in fit.names
    assert not any(n.startswith("spot_phi") for n in fit.names)                               # the spot stays in the injector's plane
    shared = CompositeFit("band_spray", comps, [G], {c: (1.0, 1.0) for c in G.channels}, "constant", 225.0, shared_toroidal=True, vary_peak=True)
    assert "phi_peak" in shared.names and "decay" in shared.names                              # the band still moves; the spot does not
    values = {**{f"band_{k}": v for k, v in PSI_TRUE.items() if k != "decay"}, "band_decay": 0.5, "band_phi_source": 200.0,
              **{f"spot_{k}": v for k, v in SPRAY_TRUE.items() if k != "decay"}, "spot_decay": 0.5, "a_band": 1.0, "a_spot": 0.5}
    fit._apply(values)
    assert t.info["phi_source"] == 225.0
    from emis3d.radDist import describe_parametric

    assert "spot (spray_spot)" in describe_parametric(fit, values) and "(at R" in describe_parametric(fit, values)


def test_spray_spot_needs_an_injector_and_runs_through_emis3d(rings, tmp_path):
    from emis3d.emis3D import Emis3D
    from emis3d.geometry import matrix_path
    from test_calibration import write_workspace

    base = {"BOLOMETERS": {"A": {"dataFileName": "a.h5"}}, "shot": 1, "solver": "parametric"}
    with pytest.raises(ValueError, match="needs injector"):
        RunConfig.from_dict({**base, "parametric": {"models": ["spray_spot"], "equilibrium": "g"}})
    run = write_workspace(tmp_path, tstart=10.0, radDistDirectories_LocIndependent=[], solver="parametric", scale_def="constant",
                          injector="SPI225", vary_peak_rad_location=True,
                          parametric={"models": ["spray_spot"], "matrices": ["low"], "equilibrium": mock.EQ_FILE, "maxiter": 3, "substeps": 1})
    ws = Workspace.resolve(workspace=tmp_path, env={}, user_config=tmp_path / "none.yaml")
    matrix_for(rings).save(matrix_path(ws, "MOCK", "low"))
    e = Emis3D(runConfigName=str(run))
    e._perform_fits()
    res = e.bestFits[10.0]["parametric"]
    assert res.model == "spray_spot" and {"spray_s", "spray_offset_deg"} <= set(res.values) and "phi_source" not in res.values

# A toroidal floor under the peak (toroidal_floor)
# ---------------------------------------------------------------------------

def test_parametric_floor_is_an_exactly_solved_column(rings):
    _, truth, template = rings
    G = synthetic_G(truth)
    shape = {k: TRUE[k] for k in ("startR", "startZ", "sigma_R", "sigma_z", "rotationAngle")}
    probe = ParametricFit("ring", template, [G], {c: (1.0, 1.0) for c in G.channels}, "exponential", 225.0, toroidal_floor=True)
    assert probe.amp_names == ["a", "c"] and probe.amp_bounds("c")[0] == 0.0
    vals = {**shape, "b": 1.0}
    X = probe.design(G, vals)
    rows = probe._rows[id(G)][0]
    np.testing.assert_allclose(np.column_stack([G.signals(c)[rows] for c in probe.voxel_columns(G, vals)]), X, rtol=1e-12)
    signal = X[:, 0] + 0.25 * X[:, 1]                                                       # a floor a quarter of the peak
    sigma = 0.02 * signal.max() * np.ones_like(signal)
    data = {c: (float(s), float(e)) for c, s, e in zip(G.channels, signal, sigma, strict=True)}
    fit = ParametricFit("ring", template, [G], data, "exponential", 225.0, fixed=shape, toroidal_floor=True)
    res = fit.search(seed=0, maxiter=20)
    assert res.values["c"] / res.values["a"] == pytest.approx(0.25, abs=0.02) and res.values["b"] == pytest.approx(1.0, abs=0.1)
    without = ParametricFit("ring", template, [G], data, "exponential", 225.0, fixed=shape).search(seed=0, maxiter=20)
    assert without.chi2 > 10 * max(res.chi2, 1e-9) and res.aic < without.aic

    none = {c: (float(s), float(e)) for c, s, e in zip(G.channels, X[:, 0], sigma, strict=True)}
    res0 = ParametricFit("ring", template, [G], none, "exponential", 225.0, fixed=shape, toroidal_floor=True).search(seed=0, maxiter=20)
    assert res0.values["c"] < 0.02 * res0.values["a"] and not any(f.startswith("c ") for f in res0.at_bounds)   # zero is an answer, not a flag


def test_floor_only_for_peaked_axisymmetric_profiles(rings, helical_setup):
    _, _, template = rings
    G, _, spot = helical_setup
    assert "c" not in ParametricFit("ring", template, [G], {c: (1.0, 1.0) for c in G.channels}, "constant", 225.0, toroidal_floor=True).amp_names
    assert "c" not in ParametricFit("helical", spot, [G], {c: (1.0, 1.0) for c in G.channels}, "exponential", 225.0, toroidal_floor=True).amp_names


def test_composite_floor_on_its_axisymmetric_components(rings, helical_setup):
    from emis3d.parametric import CompositeFit

    _, _, ring = rings
    G, _, spot = helical_setup
    fit = CompositeFit("core_spot", [("core", "ring", ring, None, None), ("spot", "helical", spot, None, None)], [G],
                       {c: (1.0, 1.0) for c in G.channels}, "exponential", 225.0, toroidal_floor=True)
    assert fit.amp_names == ["a_core", "c_core", "a_spot"] and fit.amp_bounds("c_core")[0] == 0.0


def test_floor_with_a_spray_spot_component(rings, helical_setup, spray_template):
    from emis3d.parametric import CompositeFit

    _, _, ring = rings
    G, _, _ = helical_setup
    t, _ = spray_template
    fit = CompositeFit("core_spray", [("core", "ring", ring, None, None), ("spot", "spray_spot", t, None, None)], [G],
                       {c: (1.0, 1.0) for c in G.channels}, "exponential", 225.0, toroidal_floor=True, vary_peak=True)
    assert fit.amp_names == ["a_core", "c_core", "a_spot"]                                 # the spray spot takes no floor
    assert not any(n.startswith("spot_phi") for n in fit.names)                            # and stays in the injector's plane
