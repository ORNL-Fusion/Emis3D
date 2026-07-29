"""
tests/test_raddist_fitting.py
=============================
Unit tests for main/radDistFitting.py (data mapping / parameter creation)
and main/Globals.py (path constants).
No cherab, raysect, or real file I/O required beyond what we fabricate in-memory.

NOTE: TO TEST THIS STUFF, go to the main directory, then type:
python -m pytest tests/
"""

import json
import os

import numpy as np
import pytest

from main.Globals import (
    EMIS3D_PARENT_DIRECTORY,
    EMIS3D_TOKMAK_DIRECTORY,
    SUPPORTED_TOKAMAKS,
)
from main.radDistFitting import RadDistFitting
from main.Util import find_max_nested_lists

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------


class TestGlobals:

    def test_file_path_is_directory(self):
        assert os.path.isdir(
            EMIS3D_PARENT_DIRECTORY
        ), f"EMIS3D_PARENT_DIRECTORY '{EMIS3D_PARENT_DIRECTORY}' is not a directory"

    def test_parent_directory_exists(self):
        assert os.path.isdir(EMIS3D_PARENT_DIRECTORY)

    def test_tokamak_directory_exists(self):
        assert os.path.isdir(EMIS3D_TOKMAK_DIRECTORY)

    def test_supported_tokamaks_is_list(self):
        assert isinstance(SUPPORTED_TOKAMAKS, list)
        assert len(SUPPORTED_TOKAMAKS) > 0

    def test_diii_d_is_supported(self):
        assert "DIII-D" in SUPPORTED_TOKAMAKS


# ---------------------------------------------------------------------------
# RadDistFitting (no real data; we construct a minimal JSON fixture)
# ---------------------------------------------------------------------------


def _minimal_raddist_json(
    tmp_path,
    units="Power",
    distType="elongatedRing",
    n_channels=4,
    n_bolos=2,
    n_segments=1,
):
    """
    Write a minimal radDist JSON file that RadDistFitting can load without
    needing cherab or actual measurement files.

    n_segments controls the number of toroidal observation segments per channel
    in the new format.
    """
    emission = "elongatedRing"
    channel_tags = [f"CH{i+1:02d}" for i in range(n_channels)]

    bolo_names = [f"BOLO_{b}" for b in range(n_bolos)]
    channel_order = {}
    power_data = {}
    power_error = {}
    phi_loc = {}

    for bolo in bolo_names:
        channel_order[bolo] = channel_tags
        power_data[bolo] = [[float(i + 1)] * n_segments for i in range(n_channels)]
        power_error[bolo] = [[0.1 * (i + 1)] * n_segments for i in range(n_channels)]
        phi_loc[bolo] = [
            [0.5 * (i + 1) + 0.1 * s for s in range(n_segments)]
            for i in range(n_channels)
        ]

    data = {
        "info": {
            "distType": distType,
            "emissionNames": [emission],
            "injectionLocation": 120.0,
            "tokamakName": "DIII-D",
            "units": units,
            "sigma_R": 0.1,
            "sigma_z": 1.5,
            "rotationAngle": 0.0,
            "startR": 2.0,
            "startZ": 0.0,
            "eqFileName": None,
            "saveRunsDirectoryName": "test",
        },
        "data": {
            units: {
                "channelOrder": channel_order,
                emission: {bolo: power_data[bolo] for bolo in bolo_names},
            },
            f"{units}_error": {
                emission: {bolo: power_error[bolo] for bolo in bolo_names},
            },
            "observed_phi_loc": {
                emission: {bolo: phi_loc[bolo] for bolo in bolo_names},
            },
            "toroidalRadiatedPower": {
                emission: {
                    "phi_array": list(np.linspace(0, 2 * np.pi, 20)),
                    "P_pol": [1.0] * 20,
                    "P_total": 1.0,
                }
            },
        },
    }

    path = tmp_path / "test_raddist.json"
    path.write_text(json.dumps(data))
    return str(path), bolo_names, channel_tags, emission


class TestRadDistFitting:

    def test_loads_without_error(self, tmp_path):
        path, *_ = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        assert rdf.info is not None
        assert "injectionLocation" in rdf.info

    def test_emission_names_loaded(self, tmp_path):
        path, *_ = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        assert "elongatedRing" in rdf.info["emissionNames"]

    def test_data_maps_created(self, tmp_path):
        path, bolo_names, channel_tags, emission = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        assert emission in rdf.data_maps
        for key in ("observed_phi_loc", "data", "data_error"):
            assert key in rdf.data_maps[emission], f"Missing key '{key}' in data_maps"

    def test_channel_keys_in_data_maps(self, tmp_path):
        path, bolo_names, channel_tags, emission = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        for ch in channel_tags:
            assert ch in rdf.data_maps[emission]["data"], f"Channel {ch} missing"

    def test_prepare_for_fits_creates_fitSynthetic(self, tmp_path):
        path, bolo_names, channel_tags, emission = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        channel_order = [channel_tags]  # one group with all channels
        rdf.prepare_for_fits(channel_order, data_max=None)
        assert hasattr(rdf, "fitSynthetic")
        assert emission in rdf.fitSynthetic

    def test_prepare_for_fits_data_length(self, tmp_path):
        n_ch = 4
        path, _, channel_tags, emission = _minimal_raddist_json(
            tmp_path, n_channels=n_ch
        )
        rdf = RadDistFitting(radDistPath=path)
        channel_order = [channel_tags]
        rdf.prepare_for_fits(channel_order)
        data_list = rdf.fitSynthetic[emission]["data"]
        assert len(data_list) == 1  # one bolometer group
        assert len(data_list[0]) == n_ch  # n_ch channels

    def test_data_max_scaling(self, tmp_path):
        """When data_max is provided, synthetic data should be scaled up."""
        path, _, channel_tags, emission = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        channel_order = [channel_tags]

        rdf.prepare_for_fits(channel_order, data_max=None)
        unscaled_max = find_max_nested_lists(rdf.fitSynthetic[emission]["data"])

        rdf2 = RadDistFitting(radDistPath=path)
        rdf2.prepare_for_fits(channel_order, data_max=unscaled_max * 10.0)
        scaled_max = find_max_nested_lists(rdf2.fitSynthetic[emission]["data"])

        assert scaled_max > unscaled_max

    def test_create_parameters_int_injection_location_works(self, tmp_path):
        """
        create_parameters() succeeds when injectionLocation is stored as an int.
        This also verifies the b_ parameter is generated correctly.
        """
        path, _, channel_tags, _ = _minimal_raddist_json(tmp_path)
        rdf = RadDistFitting(radDistPath=path)
        # Override to integer so the 'a_' name is valid
        rdf.info["injectionLocation"] = int(rdf.info["injectionLocation"])
        rdf.prepare_for_fits([channel_tags])
        rdf.create_parameters()

        from lmfit import Parameters

        assert isinstance(rdf.fitSynthetic["params"]["params"], Parameters)
        param_names = list(rdf.fitSynthetic["params"]["params"].keys())
        amp_params = [n for n in param_names if n.startswith("a_")]
        assert len(amp_params) >= 1
        for name in amp_params:
            assert rdf.fitSynthetic["params"]["params"][name].min >= 0.0

    def test_none_path_initialises_info_with_path_key(self):
        """RadDistFitting(None) stores {'radDistPath': None} in self.info."""
        rdf = RadDistFitting(radDistPath=None)
        assert rdf.info == {"radDistPath": None}
        assert not hasattr(rdf, "data")


# ---------------------------------------------------------------------------
# Emis3D error handling — exceptions replace error_free flag
# ---------------------------------------------------------------------------

from main.Emis3D import Emis3D


class TestEmis3DErrorHandling:

    def test_none_args_raises_value_error(self):
        """Passing no tokamakName/runConfigName must raise, not silently fail."""
        with pytest.raises(ValueError, match="required"):
            Emis3D(tokamakName=None, runConfigName=None)

    def test_missing_config_file_raises(self, tmp_path):
        """A missing run config file raises FileNotFoundError."""
        e = Emis3D(initialize=False)
        with pytest.raises(FileNotFoundError):
            e._load_config_file(tokamakName="DIII-D", runConfigName="nonexistent.yaml")

    def test_no_error_free_attribute(self):
        """error_free flag has been removed — Emis3D must not have it."""
        e = Emis3D(initialize=False)
        assert not hasattr(
            e, "error_free"
        ), "error_free should be removed; use exceptions instead"

    def test_verbose_sets_debug_logging(self):
        """verbose=True must configure the 'main' logger to DEBUG."""
        import logging

        Emis3D(initialize=False, verbose=True)
        assert logging.getLogger("main").level == logging.DEBUG

    def test_missing_bolometers_key_raises(self):
        """_load_bolometer_data raises RuntimeError when BOLOMETERS missing from info."""
        e = Emis3D(initialize=False)
        e.info = {"tokamakName": "DIII-D"}  # no BOLOMETERS key
        with pytest.raises(RuntimeError, match="BOLOMETERS"):
            e._load_bolometer_data()


# ---------------------------------------------------------------------------
# saveRadDist consolidation — base class + _folder_suffix hook
# ---------------------------------------------------------------------------


class TestSaveRadDistConsolidation:

    def test_base_class_has_save_rad_dist(self):
        """saveRadDist must now live on RadDist, not be abstract."""
        from main.radDist import RadDist
        import inspect

        # Method should be defined on RadDist itself, not just declared abstract
        assert "saveRadDist" in RadDist.__dict__
        src = inspect.getsource(RadDist.saveRadDist)
        assert "abstractmethod" not in src

    def test_subclasses_no_longer_override_save_rad_dist(self):
        """Helical, HelicalRing, ElongatedRing, SquareTube must not define saveRadDist."""
        from main.radDist import Helical, HelicalRing, ElongatedRing, SquareTube

        for cls in (Helical, HelicalRing, ElongatedRing, SquareTube):
            assert (
                "saveRadDist" not in cls.__dict__
            ), f"{cls.__name__} still overrides saveRadDist — should use base class"

    def test_helical_folder_suffix_uses_sigma_kernel(self):
        """Helical._folder_suffix must use sigmaKernel, not rotationAngle."""
        from main.radDist import Helical

        assert (
            "_folder_suffix" in Helical.__dict__
        ), "Helical must override _folder_suffix"
        import inspect

        src = inspect.getsource(Helical._folder_suffix)
        assert "sigmaKernel" in src

    def test_other_subclasses_use_base_folder_suffix(self):
        """HelicalRing, ElongatedRing, SquareTube must use the base _folder_suffix."""
        from main.radDist import HelicalRing, ElongatedRing, SquareTube

        for cls in (HelicalRing, ElongatedRing, SquareTube):
            assert (
                "_folder_suffix" not in cls.__dict__
            ), f"{cls.__name__} should not override _folder_suffix"
