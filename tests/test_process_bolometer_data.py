"""
tests/test_process_bolometer_data.py

The standalone pre-processing script (scripts/process_bolometer_data.py):
offsets, filtering, spike removal, bad-channel detection, the Emis3D-format
output, the report and the plots, on synthetic data with planted faults.
"""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from emis3d.util import read_h5

SCRIPT = Path(__file__).parents[1] / "scripts" / "process_bolometer_data.py"
spec = importlib.util.spec_from_file_location("process_bolometer_data", SCRIPT)
pbd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pbd)


def write_synthetic(folder: Path, name="CAM.h5"):
    rng = np.random.default_rng(3)
    t = np.linspace(49.9, 51.4, 6001)
    burst = np.exp(-((t - 50.95) / 0.02) ** 2)
    chans = [f"CAM_{i + 1:02d}" for i in range(12)]
    sig = np.zeros((12, t.size))
    for i in range(12):
        sig[i] = (1e7 + 2e6 * i) * burst + rng.normal(0, 2e5, t.size) + 1e6 * (i - 6)   # an offset per channel
    sig[2] = 0.0                                                                      # dead
    sig[5] = 3e5 + rng.normal(0, 1e3, t.size)                                         # flat
    sig[7, rng.integers(0, t.size, 500)] += 1.5e7                                     # spiky
    sig[9] = np.minimum(sig[9], 0.4 * sig[9].max())                                   # saturated (a long plateau)
    with h5py.File(folder / name, "w") as f:
        f.create_dataset("TIME", data=t)
        f.create_dataset("DATA_CALIBRATED", data=sig)
        f.create_dataset("channelOrder", data=[c.encode() for c in chans])
        f.create_dataset("EXTRA", data=np.arange(3))                                   # carried over unchanged
        f.attrs["shot"] = 1
    return chans, t, sig


def test_processing_finds_the_planted_faults_and_writes_emis3d_format(tmp_path, capsys):
    chans, t, sig = write_synthetic(tmp_path)
    cfg = {"files": [{"input": "CAM.h5", "camera": "CAM"}], "offset": {"window": [49.95, 50.4]},
           "filter": {"type": "savgol", "window": 9, "order": 2}, "spikes": {"remove": True}, "time_window": [50.5, 51.3],
           "bad_channels": {"manual": ["CAM_12"]}}
    (tmp_path / "process.yaml").write_text(yaml.safe_dump(cfg))
    assert pbd.main([str(tmp_path / "process.yaml"), "--no-plots"]) == 0

    out = read_h5(tmp_path / "processed" / "CAM_PROCESSED.h5")
    names = [c.decode() for c in out["channelOrder"]]
    assert names == chans and out["DATA_CALIBRATED"].shape == (12, out["TIME"].size) and list(out["EXTRA"]) == [0, 1, 2]
    assert out["TIME"][0] >= 50.5 and out["TIME"][-1] <= 51.3                           # cropped

    report = yaml.safe_load((tmp_path / "processed" / "processing_report.yaml").read_text())
    bad = report["files"][0]["bad_channels"]
    assert set(bad) == {"CAM_03", "CAM_06", "CAM_08", "CAM_10", "CAM_12"}
    assert "zero" in bad["CAM_03"][0] and "flat" in bad["CAM_06"][0] and "spiky" in bad["CAM_08"][0]
    assert "saturated" in bad["CAM_10"][0] and "manual" in bad["CAM_12"][0]
    for ch in bad:                                                                   # zeroed: Emis3D then treats them as bad
        assert np.all(out["DATA_CALIBRATED"][names.index(ch)] == 0)
    block = report["run_config_block"]["BOLOMETERS"]["CAM"]
    assert block["dataFileName"] == "CAM_PROCESSED.h5" and block["badChannels"] == sorted(bad)
    assert "badChannels" in capsys.readouterr().out

    # a good channel: offset removed (near zero before the burst) and the burst kept
    good = out["DATA_CALIBRATED"][names.index("CAM_01")]
    quiet = out["TIME"] < 50.8
    assert abs(np.mean(good[quiet])) < 5e4 and good.max() > 8e6
    offsets = report["files"][0]["offsets"]
    assert offsets["CAM_01"] == pytest.approx(-6e6, rel=0.05)


def test_dry_run_writes_nothing_and_plots_are_saved(tmp_path):
    write_synthetic(tmp_path)
    cfg = {"files": [{"input": "CAM.h5", "camera": "CAM"}], "offset": {"window": [49.95, 50.4]}}
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(cfg))
    assert pbd.main([str(tmp_path / "p.yaml"), "--dry-run", "--no-plots"]) == 0
    assert not (tmp_path / "processed").exists()
    assert pbd.main([str(tmp_path / "p.yaml")]) == 0
    assert (tmp_path / "processed" / "plots" / "CAM_overview.png").is_file()
    assert (tmp_path / "processed" / "plots" / "CAM_channels.png").is_file()


def test_per_file_overrides_and_config_errors(tmp_path):
    write_synthetic(tmp_path, "A.h5")
    write_synthetic(tmp_path, "B.h5")
    cfg = {"files": [{"input": "A.h5", "camera": "A"}, {"input": "B.h5", "camera": "B", "bad_channels": {"action": "keep"}}],
           "offset": {"window": [49.95, 50.4]}}
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(cfg))
    assert pbd.main([str(tmp_path / "p.yaml"), "--no-plots"]) == 0
    a = read_h5(tmp_path / "processed" / "A_PROCESSED.h5")["DATA_CALIBRATED"]
    b = read_h5(tmp_path / "processed" / "B_PROCESSED.h5")["DATA_CALIBRATED"]
    assert np.all(a[5] == 0) and not np.all(b[5] == 0)                                 # B keeps its flat channel
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump({"files": [{"input": "A.h5"}], "offsets": {}}))
    with pytest.raises(ValueError, match="unknown setting"):
        pbd.main([str(tmp_path / "bad.yaml"), "--no-plots"])
    (tmp_path / "lp.yaml").write_text(yaml.safe_dump({"files": [{"input": "A.h5"}], "filter": {"type": "lowpass"}}))
    with pytest.raises(ValueError, match="cutoff_hz"):
        pbd.main([str(tmp_path / "lp.yaml"), "--no-plots"])
    (tmp_path / "lp2.yaml").write_text(yaml.safe_dump({"files": [{"input": "A.h5"}], "filter": {"type": "lowpass", "cutoff_hz": 500.0},
                                                       "resample": {"dt": 2.5e-4}}))
    assert pbd.main([str(tmp_path / "lp2.yaml"), "--no-plots"]) == 0


def test_example_config_parses():
    cfg = yaml.safe_load((SCRIPT.parent / "process_bolometer_data_example.yaml").read_text())
    assert set(cfg) <= {"input_dir", "output_dir", "files", *pbd.DEFAULTS} and len(cfg["files"]) == 2
    merged = pbd.deep_merge(pbd.DEFAULTS, {k: v for k, v in cfg.items() if k in pbd.DEFAULTS})
    assert merged["offset"]["method"] == "median" and merged["bad_channels"]["action"] == "zero"


def test_inverted_channels_and_window_filters(tmp_path):
    chans, t, sig = write_synthetic(tmp_path)
    cfg = {"files": [{"input": "CAM.h5", "camera": "CAM"}], "offset": {"window": [49.95, 50.4]},
           "invert_channels": ["CAM_02"], "filter": {"type": "hanning", "window": 21}}
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(cfg))
    assert pbd.main([str(tmp_path / "p.yaml"), "--no-plots"]) == 0
    out = read_h5(tmp_path / "processed" / "CAM_PROCESSED.h5")["DATA_CALIBRATED"]
    assert out[1].min() < -8e6 and out[0].max() > 8e6                                   # CAM_02 flipped, CAM_01 not
    assert out.shape[1] == t.size


# ---------------------------------------------------------------------------
# Edge-preserving denoising, cross-channel consistency, errors
# ---------------------------------------------------------------------------

def tv_objective(x, y, lam):
    return 0.5 * np.sum((x - y) ** 2) + lam * np.sum(np.abs(np.diff(x)))


def test_tv_denoising_is_optimal_and_keeps_edges():
    from scipy.signal import savgol_filter

    rng = np.random.default_rng(0)
    truth = np.where(np.arange(2000) < 1000, 0.0, 1.0)                                  # a step
    y = truth + 0.1 * rng.normal(size=truth.size)
    lam = 0.3
    x = pbd.tv_denoise_1d(y, lam)
    assert tv_objective(x, y, lam) < tv_objective(y, y, lam)
    assert tv_objective(x, y, lam) < tv_objective(savgol_filter(y, 51, 2), y, lam)
    assert tv_objective(x, y, lam) < tv_objective(truth, y, lam) + 1e-9 or np.abs(x - truth).max() < 0.1
    np.testing.assert_allclose(pbd.tv_denoise_1d(y, 0.0), y)                             # no weight: unchanged
    assert np.ptp(pbd.tv_denoise_1d(y, 1e6)) < 1e-9 and pbd.tv_denoise_1d(y, 1e6)[0] == pytest.approx(y.mean())
    edge = slice(990, 1010)
    assert np.abs(x[edge] - truth[edge]).max() < np.abs(savgol_filter(y, 51, 2)[edge] - truth[edge]).max()   # sharper step
    assert np.std(x[:900]) < 0.3 * np.std(y[:900])                                        # noise suppressed


def test_tv_filter_through_the_script(tmp_path):
    write_synthetic(tmp_path)
    cfg = {"files": [{"input": "CAM.h5", "camera": "CAM"}], "offset": {"window": [49.95, 50.4]}, "filter": {"type": "tv", "tv_noise_factor": 2.0}}
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(cfg))
    assert pbd.main([str(tmp_path / "p.yaml"), "--no-plots"]) == 0
    out = read_h5(tmp_path / "processed" / "CAM_PROCESSED.h5")
    good = out["DATA_CALIBRATED"][0]
    assert good.max() > 8e6 and np.std(good[out["TIME"] < 50.8]) < 1e5                    # the burst kept, the noise gone
    assert out["DATA_CALIBRATED_ERROR"].shape == out["DATA_CALIBRATED"].shape and (out["DATA_CALIBRATED_ERROR"][0] > 0).all()
    assert np.median(out["DATA_CALIBRATED_ERROR"][0]) >= 0.9 * 2e5                         # at least the noise


def test_wavelet_filter_if_available(tmp_path):
    pytest.importorskip("pywt")
    rng = np.random.default_rng(1)
    y = np.where(np.arange(1024) < 512, 0.0, 1.0) + 0.1 * rng.normal(size=1024)
    x = pbd.wavelet_denoise_1d(y, "db4", None, 3.0)
    assert x.size == y.size and np.std(x[:400]) < 0.5 * np.std(y[:400])


def test_svd_flags_an_inconsistent_channel(tmp_path):
    chans, t, sig = write_synthetic(tmp_path)
    with h5py.File(tmp_path / "CAM.h5", "r+") as f:
        d = f["DATA_CALIBRATED"][()]
        d[0] = 8e6 * np.exp(-((t - 50.7) / 0.05) ** 2) + np.random.default_rng(5).normal(0, 2e5, t.size)   # bursts when nothing else does
        f["DATA_CALIBRATED"][...] = d
    cfg = {"files": [{"input": "CAM.h5", "camera": "CAM"}], "offset": {"window": [49.95, 50.4]},
           "bad_channels": {"svd": {"enabled": True, "components": 2, "residual_factor": 4.0}}}
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(cfg))
    assert pbd.main([str(tmp_path / "p.yaml"), "--no-plots"]) == 0
    bad = yaml.safe_load((tmp_path / "processed" / "processing_report.yaml").read_text())["files"][0]["bad_channels"]
    assert any("inconsistent" in r for r in bad.get("CAM_01", []))
    assert not any("inconsistent" in r for r in bad.get("CAM_11", []))                   # a normal channel is not


def test_common_mode_removal(tmp_path):
    chans, t, sig = write_synthetic(tmp_path)
    pickup = 3e5 * np.sin(2 * np.pi * 60 * t)
    with h5py.File(tmp_path / "CAM.h5", "r+") as f:
        d = f["DATA_CALIBRATED"][()]
        d[[0, 1, 3, 4, 6, 8, 10, 11]] += pickup                                           # shared by the good channels
        f["DATA_CALIBRATED"][...] = d
    time = t
    off = pbd.subtract_offset(time, d, {"window": [49.95, 50.4], "method": "median"})
    cleaned, removed = pbd.remove_common_mode(time, d - off[:, None], [49.95, 50.4], 1)
    quiet = time < 50.4
    assert np.std(cleaned[0, quiet]) < 0.8 * np.std((d - off[:, None])[0, quiet])        # the pickup is mostly gone
    assert np.corrcoef(removed[0, quiet], pickup[quiet])[0, 1] > 0.7
