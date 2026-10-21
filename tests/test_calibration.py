"""
tests/test_calibration.py

Cross-calibration (emis3d.calibration) on the built-in MOCK machine's five
cameras, with fabricated radDists and data in a temporary workspace (no ray
tracing). Each camera reads the true signal times a known miscalibration, so
the expected correction factors are exact.
"""

import json

import h5py
import numpy as np
import pytest
import yaml

from emis3d import cli
from emis3d.calibration import camera_amplitude, cross_calibrate, parse_times
from emis3d.globals import RADDIST_FORMAT
from emis3d.run_config import RunConfig
from emis3d.workspace import BUILTIN_MACHINES

MOCK_SX = BUILTIN_MACHINES / "MOCK" / "inputs" / "sxrInfo"
CAMS = ["POL_A", "POL_B", "POL_C", "SKEW", "WRAP"]
MISCAL = {"POL_A": 1.0, "POL_B": 0.8, "POL_C": 1.25, "SKEW": 1.1, "WRAP": 0.9}
TRUE_RD = 1
TIMES = np.linspace(9.0, 11.0, 201)


def cam_info(cam):
    cfg = yaml.safe_load((MOCK_SX / f"{cam}_config.yaml").read_text())
    return cfg["CHANNEL_TAGS"], np.deg2rad(cfg["CAMERA_POSITION_R_Z_PHI"][2])


def shape(i, g, n):
    """Synthetic signal of radDist i on camera g: a different profile per camera."""
    x = np.linspace(-1, 1, n)
    centre = -0.5 + 0.35 * i + 0.1 * g
    width = 0.3 + 0.05 * i
    return np.exp(-((x - centre) ** 2) / (2 * width**2)) + 0.05


def write_workspace(root, miscal=MISCAL, drift=None, scaling=None, loc_dependent=False, noise=0.01):
    (root / "emis3d.yaml").write_text("version: 1\n")
    rd_dir = root / "raddists" / "MOCK" / "ring"
    rd_dir.mkdir(parents=True)
    for i in range(3):
        phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        data = {
            "Brightness": {"channelOrder": {}, "elongatedRing": {}},
            "Brightness_error": {"elongatedRing": {}},
            "observed_phi_loc": {"elongatedRing": {}},
            "toroidalRadiatedPower": {"elongatedRing": {"phi_array": phi.tolist(), "P_pol": [1.0] * 100}},
        }
        for g, cam in enumerate(CAMS):
            tags, phi = cam_info(cam)
            data["Brightness"]["channelOrder"][cam] = tags
            data["Brightness"]["elongatedRing"][cam] = [[float(v)] for v in shape(i, g, len(tags))]
            data["Brightness_error"]["elongatedRing"][cam] = [[0.0]] * len(tags)
            data["observed_phi_loc"]["elongatedRing"][cam] = [[float(phi)]] * len(tags)
        info = {"radDistFormat": RADDIST_FORMAT, "units": "Brightness", "emissionNames": ["elongatedRing"], "injectionLocation": 0.0, "distType": "elongatedRing", "tokamakName": "MOCK"}
        (rd_dir / f"R_{i}.json").write_text(json.dumps({"info": info, "data": data}))

    rng = np.random.default_rng(0)
    meas = root / "measurements" / "MOCK" / "1"
    meas.mkdir(parents=True)
    for g, cam in enumerate(CAMS):
        tags, _ = cam_info(cam)
        m = np.full(TIMES.size, miscal[cam])
        if drift and cam in drift:
            m = m * (1 + drift[cam] * (TIMES - TIMES[0]) / (TIMES[-1] - TIMES[0]))
        clean = 1000.0 * shape(TRUE_RD, g, len(tags))[:, None] * m[None, :]
        with h5py.File(meas / f"{cam}.h5", "w") as f:
            f.create_dataset("channelOrder", data=tags)
            f.create_dataset("TIME", data=TIMES)
            f.create_dataset("DATA_CALIBRATED", data=clean * (1 + noise * rng.normal(size=clean.shape)))

    run = root / "runs" / "cal" / "run.yaml"
    run.parent.mkdir(parents=True)
    run.write_text(
        yaml.safe_dump(
            {
                "tokamakName": "MOCK",
                "shot": 1,
                "dt": 0.05,
                "radDistDirectories_LocIndependent": ["ring"],
                "radDistDirectories_LocDependent": ["ring"] if loc_dependent else [],
                "scale_def": "constant",
                "error_definition": "constant",
                "enable_dphi_scaling": False,
                "BOLOMETERS": {c: {"dataFileName": f"1/{c}.h5", "scalingFactor": (scaling or {}).get(c, 1.0)} for c in CAMS},
            }
        )
    )
    return run


# ---------------------------------------------------------------------------
# The calibration
# ---------------------------------------------------------------------------


def test_recovers_known_factors_against_a_chosen_reference(tmp_path):
    run = write_workspace(tmp_path)
    res = cross_calibrate(run, [10.0], reference="POL_A")
    for cam in CAMS:
        assert res.factor[cam] == pytest.approx(1.0 / MISCAL[cam], rel=0.01), cam
    assert set(res.best_raddist) == {f"R_{TRUE_RD}.json"}


def test_default_reference_is_the_largest_amplitude(tmp_path):
    run = write_workspace(tmp_path)
    res = cross_calibrate(run, [10.0])
    assert res.reference == "POL_C"  # reads 1.25x, the highest
    for cam in CAMS:
        assert res.factor[cam] == pytest.approx(MISCAL["POL_C"] / MISCAL[cam], rel=0.01), cam


def test_several_times_give_median_and_spread(tmp_path):
    run = write_workspace(tmp_path)
    res = cross_calibrate(run, parse_times(["9.5:10.5:0.25"]), reference="POL_A")
    assert len(res.times) == 5
    assert all(s < 0.01 for c, s in res.spread.items() if c != "POL_A")
    assert res.spread["POL_A"] == 0.0


def test_drifting_camera_shows_a_large_spread(tmp_path):
    """A factor that changes during the window means axisymmetry (or the data) is suspect."""
    run = write_workspace(tmp_path, drift={"SKEW": 0.3})
    res = cross_calibrate(run, parse_times(["9.2:10.8:0.4"]), reference="POL_A")
    assert res.spread["SKEW"] > 0.05
    assert res.spread["POL_B"] < 0.01


def test_new_scaling_factor_includes_the_current_one(tmp_path):
    """POL_B already has scalingFactor 2.0, so it reads 1.6x; the new value must still be 1/0.8."""
    run = write_workspace(tmp_path, scaling={"POL_B": 2.0})
    res = cross_calibrate(run, [10.0], reference="POL_A")
    assert res.current_scaling["POL_B"] == 2.0
    assert res.new_scaling["POL_B"] == pytest.approx(1.0 / 0.8, rel=0.01)


def test_yaml_block_is_a_valid_bolometers_section(tmp_path):
    run = write_workspace(tmp_path)
    res = cross_calibrate(run, [10.0], reference="POL_A")
    block = yaml.safe_load(res.yaml_block())["BOLOMETERS"]
    assert set(block) == set(CAMS)
    assert block["POL_B"]["dataFileName"] == "1/POL_B.h5"
    assert block["POL_B"]["scalingFactor"] == pytest.approx(1.25, rel=0.01)


def test_location_dependent_raddists_are_refused(tmp_path):
    run = write_workspace(tmp_path, loc_dependent=True)
    with pytest.raises(ValueError, match="axisymmetric"):
        cross_calibrate(run, [10.0])


def test_unknown_reference_is_an_error(tmp_path):
    run = write_workspace(tmp_path)
    with pytest.raises(ValueError, match="reference camera"):
        cross_calibrate(run, [10.0], reference="NOPE")


def test_cli_calibrate_writes_the_block(tmp_path, capsys):
    run = write_workspace(tmp_path)
    out = tmp_path / "block.yaml"
    assert cli.main(["calibrate", str(run), "--times", "10", "--reference", "POL_A", "--output", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "reference camera POL_A" in printed
    assert yaml.safe_load(out.read_text())["BOLOMETERS"]["POL_C"]["scalingFactor"] == pytest.approx(0.8, rel=0.01)


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------


def test_camera_amplitude_is_exact_and_skips_dead_channels():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    d = 2.5 * s
    d_dead = d.copy()
    d_dead[1] = 0.0  # non-positive: left out
    e = np.full(4, 0.1)
    e_dead = e.copy()
    e_dead[3] = 1.0e8  # DEAD_CHANNEL_ERROR: left out
    c, sc, chi2, n = camera_amplitude(d, e, s)
    assert c == pytest.approx(2.5) and chi2 == pytest.approx(0.0) and n == 4
    c, _, _, n = camera_amplitude(d_dead, e_dead, s)
    assert c == pytest.approx(2.5) and n == 2


def test_parse_times():
    assert parse_times(["1:2:0.5"]) == [1.0, 1.5, 2.0]
    assert parse_times(["3", "1:1.2:0.1"]) == pytest.approx([3.0, 1.0, 1.1, 1.2])
    with pytest.raises(ValueError):
        parse_times(["1:2"])


def test_crosscalib_flag_is_gone():
    """crossCalib is no longer a run-config option; cross-calibrate with `emis3d calibrate`."""
    base = {"BOLOMETERS": {}, "shot": 1, "radDistDirectories_LocIndependent": ["x"]}
    with pytest.raises(ValueError, match="crossCalib"):
        RunConfig.from_dict({**base, "crossCalib": False})


def test_residual_rejects_the_old_bolometer_names_argument():
    import emis3d.util_emis3D as U

    with pytest.raises(TypeError, match="cross_calibrate"):
        U.residual(None, None, {"emissionNames": []}, "constant", ["POL_A"])
