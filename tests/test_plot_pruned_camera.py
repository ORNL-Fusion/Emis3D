"""
tests/test_plot_pruned_camera.py

A camera of the run config that is not in the fit: none of its channels is
in the radDists (they were built before the camera was added, say), so it
is dropped when the radDists are loaded. The best-fit figure still asked
for its panel and stopped with KeyError: '<camera>'. And because
pipeline.run drew the figures before saving the fits, the fits were lost
with it.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import emis3d.emis3D as emis3D_module
from emis3d.emis3D import Emis3D
from emis3d.pipeline import run
from test_calibration import CAMS, write_workspace

T = 10.0


def without_camera(raddist_dir, camera):
    """Rewrite every radDist under raddist_dir as if it had been built without `camera`."""
    for f in raddist_dir.rglob("*.json"):
        saved = json.loads(f.read_text())
        units = saved["info"]["units"]
        for block in (saved["data"][units], saved["data"][f"{units}_error"], saved["data"]["observed_phi_loc"]):
            for per_camera in block.values():
                if isinstance(per_camera, dict):
                    per_camera.pop(camera, None)
        f.write_text(json.dumps(saved))


def test_only_cameras_in_the_fit_get_a_panel(tmp_path, caplog):
    run_config = write_workspace(tmp_path, tstart=T)
    without_camera(tmp_path / "raddists" / "MOCK" / "ring", "WRAP")
    e = Emis3D(runConfigName=str(run_config))
    assert "WRAP" in e.info["BOLOMETERS"] and "WRAP" not in e.channel_order["bolometer_order"]   # configured, but not fitted
    e._prepare_fits(evalTime=T)
    with caplog.at_level("WARNING", logger="emis3d.emis3D"):
        cameras = e._plotted_cameras(T)
    assert cameras == [c for c in CAMS if c != "WRAP"]                         # the config's order, without WRAP
    assert all(c in e.fitData[T]["boloData"] for c in cameras)
    assert any("WRAP" in r.getMessage() and "no channels in the fit" in r.getMessage() for r in caplog.records)


def test_every_camera_gets_a_panel_when_all_are_fitted(tmp_path, caplog):
    e = Emis3D(runConfigName=str(write_workspace(tmp_path, tstart=T)))
    e._prepare_fits(evalTime=T)
    with caplog.at_level("WARNING", logger="emis3d.emis3D"):
        assert e._plotted_cameras(T) == CAMS
    assert not caplog.records


def test_run_saves_the_fits_before_drawing_the_figures(monkeypatch):
    calls = []

    class Fake:
        def __init__(self, runConfigName=None, workspace=None):
            self.info = {}

        def _perform_fits(self, **kwargs):
            calls.append("fit")

        def close(self):
            calls.append("close")

        def _save_checkpoint(self, t):
            calls.append("checkpoint")

        def _save_bestFits(self):
            calls.append("save")

        def _plot_bestFit(self, save=False):
            calls.append("plot")
            raise KeyError("WRAP")                                               # any failure while drawing

    monkeypatch.setattr(emis3D_module, "Emis3D", Fake)
    with pytest.raises(KeyError):
        run("run.yaml")
    assert calls == ["fit", "close", "save", "plot"]                           # the fits were saved first

    calls.clear()
    with pytest.raises(KeyError):
        run("run.yaml", save=False)
    assert calls == ["fit", "close", "plot"]


@pytest.mark.cherab
def test_figure_is_drawn_without_the_unfitted_camera(tmp_path):
    from emis3d.mock import selftest

    selftest(workdir=tmp_path, pixel_samples=20, verbose=False)
    without_camera(tmp_path / "raddists" / "MOCK" / "ring", "WRAP")
    e = Emis3D(runConfigName=str(tmp_path / "runs" / "selftest" / "run.yaml"))
    e._perform_fits(evalTime=T)
    e._plot_bestFit(evalTime=T, save=True)                                     # used to raise KeyError: 'WRAP'
    assert list((tmp_path / "runs" / "selftest" / "results").glob("*/images/1_10.0000.png"))
    assert plt.get_fignums() == []
