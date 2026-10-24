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
    assert a == pytest.approx(TRUE["a"], rel=1e-6) and chi2 < 1e-12            # the truth fits exactly
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


def test_run_config_parametric_block():
    cfg = RunConfig.from_dict({**RUN, "parametric": {"model": "ring", "matrices": ["low", "medium"], "bounds": {"b": [0, 2]}}})
    assert cfg.parametric["model"] == "ring" and cfg.radDistDirectories_LocIndependent == []
    with pytest.raises(ValueError, match="No radDist directories"):
        RunConfig.from_dict(RUN)
    with pytest.raises(ValueError, match="parametric.model"):
        RunConfig.from_dict({**RUN, "parametric": {"model": "banana"}})
    with pytest.raises(ValueError, match="Unknown parametric options"):
        RunConfig.from_dict({**RUN, "parametric": {"model": "ring", "iterations": 3}})
    with pytest.raises(ValueError, match="does not have"):
        RunConfig.from_dict({**RUN, "parametric": {"model": "ring", "bounds": {"psi_center": [0, 1]}}})
    with pytest.raises(ValueError, match="equilibrium"):
        RunConfig.from_dict({**RUN, "parametric": {"model": "shell"}})
    assert set(MODELS) == {"ring", "tube", "shell"}


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
                                   "parametric": {"model": "ring", "matrices": ["low"], "maxiter": 40, "bounds": {"rotationAngle": [0, 90]}},
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

    run = write_workspace(tmp_path, tstart=10.0, radDistDirectories_LocIndependent=[],
                          parametric={"model": "ring", "matrices": ["nope"]})
    msgs = [f.message for f in check(run) if f.level == "error"]
    assert any("geometry matrix 'nope' not found" in m and "emis3d build-matrix MOCK --resolution nope" in m for m in msgs)
