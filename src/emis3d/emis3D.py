# -*- coding: utf-8 -*-
"""
Main solving routine. The radDist data will be loaded and organized
based on how the program observed each bolometer:

------------------------------------------------------------------------------------------------
synthetic = [
        [ [sig1_1, sig1_2, ..], [sig2_1, sig2_2, ...], ... ],   # injection location 1
        [ [sig1_1, sig1_2, ..], [sig2_1, sig2_2, ...], ... ],   # injection location 2
        [ [sig1_1, sig1_2, ..], [sig2_1, sig2_2, ...], ... ],   # injection location 3,
        ...
]

data = [
        [ [sig1_1, sig1_2, ..], [sig2_1, sig2_2, ...], ... ] # nested list of each bolometer array
]

data_err = [
        [ [sig1_1, sig1_2, ..], [sig2_1, sig2_2, ...], ... ] # nested list of each bolometer array
]

scale = [
           [s1, s2, ...], # Scaling factor for each bolometer based on injection location 1
           [s1, s2, ...], # Scaling factor for each bolometer based on injection location 2
]

This scale array is purely optional and will be set to 1 if not specified. It is used for
some scaling function definitions (such as distance from the injection location for the
helical radDist).
------------------------------------------------------------------------------------------------



The overall minimization function is organized as such:
------------------------------------------------------------------------------------------------
res = ((data - scale_function(params, scale) * synthetic) / error)

If you have multiple injection locations:
res = ((data - scale_function(params_1, scale_1) * synthetic_1) / error) +
      ((data - scale_function(params_2, scale_2) * synthetic_2) / error) + ...

The LMFIT minimzation routine takes care of squaring the residual, that is why we don't do it here
------------------------------------------------------------------------------------------------

Re-organised during the refactor - JLH Aug., 2025


REMINDERS:
1. Make sure that the pre-processed SXR/Bolometer data are in the same units as those when the radDists were created
2. The bolometers have different responsivities with respect to each other! Some pre-analysis of the bolometer data is necessary
in order to scale them relative to each other. Then you should load the processed data in this program.
3. This program uses a right-handed coordinate system (positive phi in counter-clockwise direction when looking down).
So you need to offset your angles by 360 - x from DIII-D coordinates.
4. Cross-calibration between cameras is a separate tool: `emis3d calibrate` (emis3d.calibration).
This is typically done during the current quench, when the radiation is assumed to be axisymmetric. The flag allows for each
bolometer to have indpendent scaling factors, see the example on how to cross-calibrate uncalibrated bolometers.


TODO:
1. Prepare fits -> Write definition to combine multiple locationDependent values together,
see _combine_synthetics_for_fits as a older starting point
2. Give the user the option to use the new error technique or use the error from the data
3. Double check field line tracer with output from MOFAT
4. Add checker to account for different phi values when calculating the total power in _post_process_calculations

Biggest Issues:
1. Find good radDist functions that represent what is going on
    - Add a tomography radDist mapping function (like BOLT?)


BUG:
1. Is the total radiated power calculated correctly?

"""

import os
import difflib
import logging
import shutil
import time
import warnings
from pathlib import Path

import dill
import yaml
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, report_fit

import emis3d.util_emis3D as Util_emis3D
import emis3d.uncertainty as uncertainty

logger = logging.getLogger(__name__)
from dataclasses import asdict
from typing import Any
from emis3d.run_config import RunConfig
from emis3d.workspace import Workspace, set_workspace
from emis3d.tokamak import Tokamak
from emis3d.util import (
    config_loader,
    convert_arrays_to_list,
    extract_end_numbers,
    find_max_nested_lists,
    get_filenames_in_directory,
    read_h5,
)
from emis3d.radDistFitting import RadDistFitting
from emis3d.radDist import Helical, ElongatedRing, HelicalRing

# --- Fit weight assigned to dead / zero-signal channels so the minmizer ignores
# them.

# --- Plot styling
PLOT_COLORS = ["green", "orangered", "blue", "cyan", "magenta"]
PLOT_MARKERS = ["^", "o", "s", "D", "v"]
UNIT_LABELS = {"Power": "[W]", "Radiance": "[W / (m2 sr)]", "Brightness": "[W / m2]"}


class Emis3D:

    def __init__(
        self,
        tokamakName: str | None = None,
        runConfigName: str | None = None,
        initialize: bool = True,
        verbose: bool = False,
        workspace: str | os.PathLike | Workspace | None = None,
    ):
        """
        Main class for running the emis3D program.

        Parameters
        ----------
        tokamakName   : Name of the tokamak (optional if the run config has 'tokamakName')
        runConfigName : Path to a run configuration file, or its name inside the
                        workspace's 'runs' location
        initialize    : When True the standard initialisation pipeline is run
        verbose       : Print extra progress information
        workspace     : Workspace folder or Workspace object; otherwise found from
                        the run config's location, EMIS3D_RUNS_DIR or the
                        user config (see emis3d.workspace)
        """

        # --- Initialize variables
        self.data = {}
        self.info: dict[str, Any] = {}  # filled from the run config by _load_config_file
        self.verbose = verbose
        self._workspace_arg = workspace
        self.ws: Workspace | None = workspace if isinstance(workspace, Workspace) else None
        self.run_config_path: Path | None = None
        self._results_path: Path | None = None

        # Configure logging level for the whole 'main' package
        if verbose:
            logging.getLogger("emis3d").setLevel(logging.DEBUG)

        if initialize:
            self._initialize(tokamakName=tokamakName, runConfigName=runConfigName)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self, tokamakName: str | None = None, runConfigName: str | None = None) -> None:
        """Load configuration, bolometer data and radDists."""
        if runConfigName is None:
            raise ValueError("runConfigName is required to initialise Emis3D")

        Util_emis3D.print_intro()
        self._load_config_file(tokamakName=tokamakName, runConfigName=runConfigName)
        self._load_bolometer_data()
        self._create_master_channel_order()
        self._load_radDists()
        self._prune_channel_order_to_radDists()

    def _perform_fits(self, evalTime: float) -> None:
        """
        Prepares the data, preforms the fits

        Parameters
        ----------
        evalTime   : Time to run the fit
        """

        # Added a bunch of timing flags, to see where improvement can be made
        self._prepare_fits(evalTime=evalTime)
        t_start = time.time()

        self._minimize_radDists(evalTime=evalTime)
        logger.info(f"→ Fitting done in {time.time() - t_start:.2f} seconds")
        t_start = time.time()
        self._post_process_fit_arrangement(evalTime=evalTime)

        logger.info(f"→ Done with fit post-processing step 1 out of 2 in {time.time() - t_start:.2f} seconds")
        t_start = time.time()

        self._post_process_bolo_locations(evalTime=evalTime)
        self._post_process_calculations(evalTime=evalTime)
        self._post_process_uncertainty(evalTime=evalTime)
        logger.info(f"→ Done with fit post-processing step 2 out of 2 in {time.time() - t_start:.2f} seconds")

    # ------------------------------------------------------------------
    # Configuration / data loading
    # ------------------------------------------------------------------

    def _load_config_file(
        self,
        tokamakName: str | None = None,
        runConfigName: str | None = None,
        pathFileName: str | os.PathLike | None = None,
    ) -> None:
        """
        Loads the YAML configuration file for the given tokamak

        Either supply ``pathFileName`` directly, or supply both
        ``tokamakName`` and ``runConfigName`` to have the path constructed
        automatically.
        """

        ws_arg = self._workspace_arg
        if pathFileName is not None:
            config_path = Path(pathFileName).expanduser()
        elif runConfigName is None:
            raise ValueError("runConfigName or pathFileName is required to load a config file")
        elif Path(str(runConfigName)).expanduser().is_file():
            config_path = Path(str(runConfigName)).expanduser()
        else:
            # --- A name inside the workspace's runs location
            pre = ws_arg if isinstance(ws_arg, Workspace) else Workspace.resolve(workspace=ws_arg)
            config_path = pre.path("runs", tokamak=tokamakName) / runConfigName

        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        raw = config_loader(str(config_path), verbose=self.verbose)

        if raw is None:
            raise RuntimeError(f"Could not load the configuration file: {config_path}")

        # --- Validate, and fill in defaults for every configuration option
        try:
            self.config = RunConfig.from_dict(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid run config {config_path}: {e}") from e

        self.info = asdict(self.config)

        tokamakName = tokamakName or self.info.get("tokamakName")
        if tokamakName is None:
            raise ValueError("tokamakName must be given, or set in the run config")

        # --- Resolve the workspace from the run config's location and 'paths:' block
        self.run_config_path = config_path.resolve()
        if isinstance(ws_arg, Workspace):
            self.ws = ws_arg
        else:
            self.ws = Workspace.resolve(
                workspace=ws_arg,
                run_config=self.run_config_path,
                run_paths=self.info.get("paths"),
            )
        set_workspace(self.ws)  # tokamaks and radDists built later use the same locations

        self.info["tokamakName"] = tokamakName
        self.info["runConfigName"] = runConfigName

    def _load_radDists(self) -> None:
        """
        Loads each radDist within the directories listed in the config file:
        radDistDirectories_LocIndependent, and radDistDirectories_LocDependent.
        """

        if self.verbose:
            logger.debug("Loading radDists")

        # --- Initialise synthetic signal arrays
        self.data.update({"synthetic": {"locDependent": {}, "locIndependent": {}}})
        self.files = []

        if not self.info:
            raise RuntimeError("Load a run config before loading radDists")
        loc_indep = list(self.info.get("radDistDirectories_LocIndependent") or [])
        loc_dep = list(self.info.get("radDistDirectories_LocDependent") or [])
        if not (loc_indep or loc_dep):
            raise RuntimeError("No radDist folders in the run config: set " "radDistDirectories_LocIndependent and/or radDistDirectories_LocDependent")

        base = self._workspace().path("raddists", tokamak=self.info["tokamakName"])

        # --- Location independent radDists
        count_ = 0
        for dir_ in loc_indep:
            for rd in self._load_radDist_folder(base, dir_, "radDistDirectories_LocIndependent"):
                self.data["synthetic"]["locIndependent"][count_] = rd
                count_ += 1

        # --- Location dependent radDists
        for ii, dir_ in enumerate(loc_dep):
            loaded = self._load_radDist_folder(base, dir_, "radDistDirectories_LocDependent")
            self.data["synthetic"]["locDependent"][f"loc_{ii}"] = dict(enumerate(loaded))

        if self.verbose:
            logger.debug("Done loading radDists")

    @staticmethod
    def _load_radDist_folder(base: Path, dir_: str, key: str) -> list:
        """
        Every radDist under base/dir_. Raises if the folder is missing or
        nothing in it loads; skips (with a warning) individual files that fail.
        """
        folder = base / dir_
        if not folder.is_dir():
            available = sorted(str(p.relative_to(base)) for p in base.glob("**/") if p != base) if base.is_dir() else []
            close = difflib.get_close_matches(str(dir_), available, n=3, cutoff=0.5)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            listing = f" Folders in {base}: {available[:10]}" if available else f" ({base} does not exist)"
            raise FileNotFoundError(f"radDist folder '{dir_}' (from {key}) not found: {folder}.{hint}{listing}")

        files = get_filenames_in_directory(str(folder))
        if not files:
            raise FileNotFoundError(f"radDist folder '{dir_}' (from {key}) has no radDist files: {folder}")

        loaded, failures = [], []
        for file_ in files:
            try:
                rd = RadDistFitting(radDistPath=file_)
                if not rd._load_ok:
                    raise ValueError(rd.load_error or "could not be read")
                loaded.append(rd)
            except Exception as e:
                failures.append(f"{file_}: {e}")
                warnings.warn(f"Skipping radDist {file_}: {e}", stacklevel=2)
                logger.warning("Skipping radDist %s: %s", file_, e)

        if not loaded:
            shown = "\n  ".join(failures[:3])
            raise RuntimeError(f"None of the {len(files)} files in radDist folder '{dir_}' loaded. First errors:\n  {shown}")
        logger.info("Loaded %d radDists from %s (%d skipped)", len(loaded), folder, len(failures))
        return loaded

    def _load_bolometer_data(self) -> None:
        """
        Load pre-calibrated SXR/bolometer data from the workspace's
        'measurements' location. ``dataFileName`` in the run config is relative
        to it (or absolute).
        """

        if not self.info or "BOLOMETERS" not in self.info:
            raise RuntimeError("No BOLOMETERS found in the config file")

        self.data["observed"] = {}
        for bolo_ in self.info["BOLOMETERS"]:
            pathFileName = str(
                self._workspace().path(
                    "measurements",
                    tokamak=self.info["tokamakName"],
                    shot=self.info.get("shot"),
                )
                / self.info["BOLOMETERS"][bolo_]["dataFileName"]
            )
            if not os.path.isfile(pathFileName):
                raise FileNotFoundError(f"Bolometer data file not found: {pathFileName}")

            logger.debug("Loading bolometer data: %s", pathFileName)

            temp_ = read_h5(pathFileName)

            for ii, ch_ in enumerate(temp_["channelOrder"]):
                temp_["channelOrder"][ii] = ch_.decode("utf-8")

            temp_["DATA_CALIBRATED"] *= self.info["BOLOMETERS"][bolo_].get("scalingFactor", 1.0)

            # --- Bad channels: listed under badChannels, or zero for the whole
            # shot (the DIII-D saver zeroes channels it was told are bad)
            chans = list(temp_["channelOrder"])
            listed = set(self.info["BOLOMETERS"][bolo_].get("badChannels", []))
            unknown = listed - set(chans)
            if unknown:
                raise ValueError(f"badChannels for {bolo_} are not in its data file: {sorted(unknown)}")
            zero = {ch for ch, row in zip(chans, np.asarray(temp_["DATA_CALIBRATED"])) if np.all(row == 0)}
            if zero - listed:
                logger.warning(
                    "%s: channels %s are zero for the whole shot and are treated as bad; " "list them under badChannels to silence this",
                    bolo_,
                    sorted(zero - listed),
                )
            temp_["badChannels"] = listed | zero

            self.data["observed"][bolo_] = temp_

    def _create_master_channel_order(self) -> None:
        """
        Build a master channel list from the loaded bolometer data.
        All radDists and raw data will be ordered to match this list.

        List is of the format:
        [[bolo1_1, bolo1_2, ...], [bolo2_1, bolo2_2, ...], ...]
        """

        if not self.info or "BOLOMETERS" not in self.info:
            raise RuntimeError("No BOLOMETERS found in the config file")

        self.channel_order = {
            "bolometer_order": [],
            "channel_list": [],
        }

        for bolo in self.data["observed"]:
            self.channel_order["bolometer_order"].append(bolo)
            self.channel_order["channel_list"].append(list(self.data["observed"][bolo]["channelOrder"]))

    def _prune_channel_order_to_radDists(self) -> None:
        """
        Delete master-order channels that are absent from the radDist synthetic
        data, so nothing is passed to the minimizer for them.
        """

        if not hasattr(self, "data") or "synthetic" not in self.data:
            return

        # --- Gather every loaded radDist
        radDists = []
        synth = self.data["synthetic"]
        for loc in synth.get("locDependent", {}):
            for number_ in synth["locDependent"][loc]:
                radDists.append(synth["locDependent"][loc][number_])
        for number_ in synth.get("locIndependent", {}):
            radDists.append(synth["locIndependent"][number_])

        if not radDists:
            return

        # --- Channels available in EVERY radDist (intersection)
        available = None
        for radD in radDists:
            if not getattr(radD, "_load_ok", True) or not hasattr(radD, "data_maps"):
                continue
            radD_avail = None
            for emissionName in radD.info["emissionNames"]:
                keys = set(radD.data_maps[emissionName]["data"].keys())
                radD_avail = keys if radD_avail is None else radD_avail & keys
            if radD_avail is None:
                continue
            available = radD_avail if available is None else available & radD_avail

        if available is None:
            return

        # --- Prune the master order, dropping emptied bolometers and keeping
        # bolometer_order parallel to channel_list.
        dropped = []
        new_channel_list = []
        new_bolometer_order = []
        for ii, bolo_chans in enumerate(self.channel_order["channel_list"]):
            kept = [ch for ch in bolo_chans if ch in available]
            dropped.extend([ch for ch in bolo_chans if ch not in available])
            if kept:
                new_channel_list.append(kept)
                new_bolometer_order.append(self.channel_order["bolometer_order"][ii])

        self.channel_order["channel_list"] = new_channel_list
        self.channel_order["bolometer_order"] = new_bolometer_order

        if dropped:
            logger.warning(
                "Dropped %d channel(s) absent from the radDist synthetic data; " "they will not be passed to the minimizer: %s",
                len(dropped),
                ", ".join(str(ch) for ch in dropped),
            )

    # ------------------------------------------------------------------
    # Fit preparation
    # ------------------------------------------------------------------

    def _average_observed_data(self, arrayName: str = "", evalTime: float | None = None) -> list:
        """
        Average SXR/bolometer data over a window [evalTime-dt, evalTime+dt].

        Returns a flat list of averaged channel values, or zeros if evalTime
        is None or dt is not configured.
        """

        if evalTime is None:
            return [0] * self.data["observed"][arrayName]["NUM_CHANNELS"]

        time_ = self.data["observed"][arrayName]["TIME"]

        if not self.info or self.info["dt"] is None:
            raise ValueError("Set dt (the averaging half-width) in the run config")

        dt_ = self.info["dt"]
        start = np.abs(time_ - (evalTime - dt_)).argmin()
        end = np.abs(time_ - (evalTime + dt_)).argmin()

        # --- Average the data, include endpoint to prevent errors when dt is too small.
        # A channel with no valid samples stays NaN and is treated as bad.
        window = np.asarray(self.data["observed"][arrayName]["DATA_CALIBRATED"][:, start : end + 1], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vals = np.nanmean(window, axis=1)

        return convert_arrays_to_list(vals)

    def _channel_noise(self, arrayName: str, evalTime: float) -> np.ndarray | None:
        """
        Per-channel noise of the averaged signal at evalTime: the standard
        deviation of the raw samples in the run config's noise_window (a quiet
        period), divided by sqrt(number of samples averaged). None without a
        noise_window.
        """
        window = self.info.get("noise_window")
        if not window:
            return None
        obs = self.data["observed"][arrayName]
        time_ = np.asarray(obs["TIME"], dtype=float)
        quiet = (time_ >= window[0]) & (time_ <= window[1])
        if quiet.sum() < 3:
            raise ValueError(f"noise_window {window} holds fewer than 3 samples for {arrayName}")
        std = np.nanstd(np.asarray(obs["DATA_CALIBRATED"], dtype=float)[:, quiet], axis=1, ddof=1)

        dt_ = self.info["dt"]
        start = np.abs(time_ - (evalTime - dt_)).argmin()
        end = np.abs(time_ - (evalTime + dt_)).argmin()
        return std / np.sqrt(end - start + 1)

    def _prepare_data_for_fit(self, evalTime: float) -> None:
        """
        Average and organise observed data for the minimization fit.

        Creates
        -------
        self.fitData[evalTime]['observed']       : Averaged calibrated data
        self.fitData[evalTime]['observed_error'] : Error estimates
        """

        if not hasattr(self, "fitData"):
            self.fitData = {}

        # NOTE:
        """
        - Observed is what is used in the minimization process, it is a set of
          nested list, the same way that self.channel_order['channel_list'] is organized
        - Bolo data is organzied to make plotting easier since it is a dict of each bolometer.
        """

        self.fitData[evalTime] = {
            "observed": [],
            "observed_error": [],
            "boloData": {},
            "boloData_error": {},
        }

        # --- Average and map the data to a dict
        data_, noise_, bad_ = {}, {}, set()
        for bolo_ in self.data["observed"]:
            obs = self.data["observed"][bolo_]
            temp = self._average_observed_data(arrayName=bolo_, evalTime=evalTime)
            data_.update(dict(zip(obs["channelOrder"], temp)))
            noise = self._channel_noise(bolo_, evalTime)
            if noise is not None:
                noise_.update(dict(zip(obs["channelOrder"], noise)))
            bad_ |= set(obs.get("badChannels", ()))

        self.fitData[evalTime]["dataMap"] = data_

        # --- The peak over the good channels sets the error floor
        good = [data_[ch] for bolo_ in self.channel_order["channel_list"] for ch in bolo_ if ch not in bad_ and np.isfinite(data_[ch])]
        max_ = max(good, default=0.0)

        for ii, bolo_chans in enumerate(self.channel_order["channel_list"]):
            temp = [data_[ch] for ch in bolo_chans]
            noise = [noise_[ch] for ch in bolo_chans] if noise_ else None
            err = Util_emis3D.channel_error(
                self.info["error_definition"],
                temp,
                max_,
                floor=self.info["error_floor"],
                relative=self.info["error_relative"],
                noise=noise,
            )
            # --- Bad or missing channels are left out of the fit
            out = np.array([ch in bad_ or not np.isfinite(data_[ch]) for ch in bolo_chans])
            err[out] = np.inf
            temp_e = err.tolist()

            self.fitData[evalTime]["observed"].append(temp)
            self.fitData[evalTime]["observed_error"].append(temp_e)

            # --- Temp is a nested list of bolometers [[Bolo1-1, Bolo1-2, ...], [Bolo2-1, Bolo2-2], etc.]
            bolo_name = self.channel_order["bolometer_order"][ii]
            self.fitData[evalTime]["boloData"][bolo_name] = temp
            self.fitData[evalTime]["boloData_error"][bolo_name] = temp_e

        if self.verbose:
            logger.debug("→ Observed data prepared for fitting")

    def _prepare_synthetic_for_fits(self, evalTime: float) -> None:
        """Prepare and parameterise every radDist for the minimization."""

        print_minor_error = True

        def print_error(radD):
            logger.info("")
            logger.warning("-" * 10 + "MINOR ERROR" + "-" * 10)
            logger.warning(f"Channels: {radD.info['ERROR CHANNELS']}were not found in the radDist, they will be ignored")
            logger.warning("-" * 31 + "\n")
            logger.info("")

        logger.debug("→ Preparing synthetic data for fitting")

        # --- Scale the synthetic data to observed for better fitting
        max_data_val = find_max_nested_lists(self.fitData[evalTime]["observed"])

        if self.info is not None:

            # --- Arrange and create parameters for the location dependent data
            for loc in self.data["synthetic"]["locDependent"]:
                for number_ in self.data["synthetic"]["locDependent"][loc]:
                    radD = self.data["synthetic"]["locDependent"][loc][number_]
                    radD.prepare_for_fits(
                        self.channel_order["channel_list"],
                        data_max=max_data_val,
                    )

                    radD.create_parameters(
                        enable_dphi_scaling=self.info["enable_dphi_scaling"],
                        vary_peak_rad_location=self.info["vary_peak_rad_location"],
                        scale_def=self.info["scale_def"],
                        peak_shift_max_degrees=self.info["peak_shift_max_degrees"],
                    )
                    if "ERROR CHANNELS" in radD.info and print_minor_error:
                        print_error(radD)
                        print_minor_error = False

            # --- Arrange and create parameters for the location independent data
            for number_ in self.data["synthetic"]["locIndependent"]:
                radD = self.data["synthetic"]["locIndependent"][number_]
                radD.prepare_for_fits(self.channel_order["channel_list"], data_max=max_data_val)
                radD.create_parameters(
                    enable_dphi_scaling=self.info["enable_dphi_scaling"],
                    vary_peak_rad_location=self.info["vary_peak_rad_location"],
                    scale_def=self.info["scale_def"],
                    peak_shift_max_degrees=self.info["peak_shift_max_degrees"],
                )

                if "ERROR CHANNELS" in radD.info and print_minor_error:
                    print_error(radD)
                    print_minor_error = False

        if self.verbose:
            logger.debug("→ Done preparing synthetic data for fit")

    def _build_synthetic_dict(self, radDist_, locDependence: str, number, location=None) -> dict:
        """
        Helper: build the synthetic_dict entry shared by both locIndependent
        and locDependent radDist loops in _prepare_fits.
        """
        if location is None:
            source = self.data["synthetic"][locDependence][number]
        else:
            source = self.data["synthetic"][locDependence][location][number]

        in_ = {}
        if radDist_.info is not None:
            in_ = radDist_.info
        else:
            raise AttributeError("Self.info should not be none!")

        synthetic_dict = {
            "info": in_,
            "paramName": radDist_.fitSynthetic["params"]["paramName"],
            "injectionLocation": source.info["injectionLocation"],
            "injectionLocation_rad": np.deg2rad(source.info["injectionLocation"]),
            "emissionNames": radDist_.info["emissionNames"],
        }

        for emissionName in radDist_.info["emissionNames"]:
            synthetic_dict[emissionName] = {
                "scaleSynth": radDist_.fitSynthetic[emissionName]["scaleSynth"],
                "observed_phi_loc": radDist_.fitSynthetic[emissionName]["observed_phi_loc"],
                "data": radDist_.fitSynthetic[emissionName]["data"],
                "data_error": radDist_.fitSynthetic[emissionName]["data_error"],
            }

            # --- Toroidal radiated-power profile, used by the helical endpoint
            # continuity constraint (Util_emis3D.helical_endpoint_penalty)
            tor_power = radDist_.data["toroidalRadiatedPower"][emissionName]
            synthetic_dict[emissionName]["phi_array"] = tor_power["phi_array"]
            synthetic_dict[emissionName]["P_pol"] = tor_power["P_pol"]

        return synthetic_dict

    def _prepare_fits(self, evalTime: float) -> None:
        """
        Organise data and synthetic signals into self.fits[evalTime] in
        preparation for _minimize_radDists.
        """

        if self.verbose:
            logger.debug("→ Preparing data for fitting")

        self._prepare_data_for_fit(evalTime=evalTime)
        self._prepare_synthetic_for_fits(evalTime=evalTime)

        if self.verbose:
            logger.debug("→ Arranging radDists for fitting")

        if not hasattr(self, "fits"):
            self.fits = {}

        # --- Initialize the fitting dictionary
        self.fits[evalTime] = {}
        fitCount = -1

        # --- Location-independent radDists
        for number in self.data["synthetic"]["locIndependent"]:
            fitCount += 1
            radDist_ = self.data["synthetic"]["locIndependent"][number]

            self.fits[evalTime][fitCount] = {
                "info": {
                    "radDists": {
                        "locationdependence": "locIndependent",
                        "radDistNumber": number,
                        "location": None,
                    }
                },
                "parameters": radDist_.fitSynthetic["params"]["params"],
                "synthetic_dict": self._build_synthetic_dict(radDist_, "locIndependent", number),
            }

        # --- Location-dependent radDists
        for loc_ in self.data["synthetic"]["locDependent"]:
            for number in self.data["synthetic"]["locDependent"][loc_]:
                fitCount += 1
                radDist_ = self.data["synthetic"]["locDependent"][loc_][number]

                self.fits[evalTime][fitCount] = {
                    "info": {
                        "radDists": {
                            "locationdependence": "locDependent",
                            "radDistNumber": number,
                            "location": loc_,
                        }
                    },
                    "parameters": radDist_.fitSynthetic["params"]["params"],
                    "synthetic_dict": self._build_synthetic_dict(radDist_, "locDependent", number, location=loc_),
                }

        num_fits = len(self.fits[evalTime])
        if self.info is not None:
            self.info["numFits"] = num_fits
        # --- Ranking score per fit (Util_emis3D.fit_score, lower is better)
        self.fits[evalTime]["scores"] = np.full(num_fits, np.inf)

    # ------------------------------------------------------------------
    # Minimization
    # ------------------------------------------------------------------

    def _minimize_radDists(self, evalTime: float) -> None:
        """
        Fit every candidate radDist in self.fits[evalTime] (Util_emis3D.fit_radDist,
        with the run config's fit_method) and record its score (fit_score: AIC).
        """

        try:
            # --- Data used for fitting
            data_dict = self.fitData[evalTime]

            # --- Soft-constraint weight tying helical endpoints together.
            # Active (1.0) by default; set to 0.0 in the run config to disable.
            helical_endpoint_weight = self.info["helical_endpoint_weight"]

            for ii in self.fits[evalTime]:

                if not isinstance(ii, int):
                    continue

                if ii % 1_000 == 0 and self.verbose:
                    logger.debug(f"→ Preforming fit {ii} out of {self.info['numFits']}")

                synth_dict = self.fits[evalTime][ii]["synthetic_dict"]
                pars = self.fits[evalTime][ii]["parameters"]

                try:
                    fit = Util_emis3D.fit_radDist(
                        pars,
                        data_dict,
                        synth_dict,
                        self.info["scale_def"],
                        helical_endpoint_weight,
                        method=self.info["fit_method"],
                    )
                    self.fits[evalTime][ii]["fit"] = fit
                    self.fits[evalTime]["scores"][ii] = Util_emis3D.fit_score(fit)
                except Exception as e:
                    logger.warning(f"An error occurred during the {ii} iteration: {e}")
                    self.fits[evalTime]["scores"][ii] = np.inf

        except Exception as e:
            logger.warning(f"An error occurred while fitting: {e}")

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _rebuild_radDist(
        self,
        evalTime: float,
        locationDependence: str = "locIndependent",
        radDistNumber: int = 0,
        location=None,
        bestFit: bool = False,
    ):
        """
        Rebuild a radDist object from stored info so it can be plotted.
        """

        if bestFit:
            if self.info is not None:
                d_ = self.bestFits[evalTime]["info"]["radDists"]
                locationDependence = d_["locationdependence"]
                radDistNumber = d_["radDistNumber"]
                location = d_["location"]

        # --- Gather the radDist information
        if location is None:
            rad_ = self.data["synthetic"][locationDependence][radDistNumber]
        else:
            rad_ = self.data["synthetic"][locationDependence][location][radDistNumber]

        info = rad_.info

        # --- Create the radDist
        if info["distType"].lower() == "helical":
            radDist_ = Helical(config=rad_.info, startR=rad_.info["startR"], startZ=rad_.info["startZ"])

        elif info["distType"].lower() == "elongatedring":
            radDist_ = ElongatedRing(config=rad_.info)
        elif info["distType"].lower() == "helicalring":
            radDist_ = HelicalRing(config=rad_.info, startR=rad_.info["startR"], startZ=rad_.info["startZ"])
        else:
            logger.info(f"_rebuild_radDist() only supports Helical, ElongatedRing, or HelicalRing")
            return None

        # --- Update the information
        if radDist_.info is not None:
            radDist_.info.update(info)

        # --- Update the data
        radDist_.data = dict(rad_.data)

        # --- Build the tokamak
        radDist_._build_tokamak(
            tokamakName=radDist_.info["tokamakName"],
            mode="Build",
            reflections=False,
            eqFileName=radDist_.info["eqFileName"],
        )

        # --- Build the field line, if it is a helical distribution
        if isinstance(radDist_, Helical) or isinstance(radDist_, HelicalRing):
            radDist_.setFieldLine()

        return radDist_

    def _post_process_fit_arrangement(self, evalTime: float) -> None:
        """Identify the best fit and reorganise synthetic data by bolometer."""

        # --- Find the best fit
        scores = np.asarray(self.fits[evalTime]["scores"], dtype=float)
        if not np.any(np.isfinite(scores)):
            raise RuntimeError(f"Every fit failed at t={evalTime}; see the warnings above for the errors")
        bestFitID = int(np.argmin(scores))

        # --- Print the results of the best fit
        print("\n-----------Best Fit-----------")
        report_fit(self.fits[evalTime][bestFitID]["fit"])
        print("------------------------------\n")

        # --- Store the best fit
        if not hasattr(self, "bestFits"):
            self.bestFits = {}

        self.bestFits[evalTime] = self.fits[evalTime][bestFitID]
        self.bestFits[evalTime]["bestFitID"] = bestFitID

        # --- First multiply the synthetic data by the fit parameters
        data_ = Util_emis3D.residual(
            self.bestFits[evalTime]["fit"].params,
            None,
            self.bestFits[evalTime]["synthetic_dict"],
            self.info["scale_def"],
            residual=False,
        )

        # --- Arrange each set in a dictionary based on the bolometer channel name
        self.bestFits[evalTime]["synthData"] = {}
        for emissionName in self.bestFits[evalTime]["synthetic_dict"]["emissionNames"]:
            self.bestFits[evalTime]["synthData"][emissionName] = {}
            temp_dict = {}
            # --- Map each data point to the correct channel
            for ii in range(len(self.channel_order["channel_list"])):
                temp_dict.update(
                    dict(
                        zip(
                            self.channel_order["channel_list"][ii],
                            data_[emissionName][ii],
                        )
                    )
                )

            # --- Loop over the bolometer and channels to build the lists
            for ii, bolo_ in enumerate(self.channel_order["bolometer_order"]):
                self.bestFits[evalTime]["synthData"][emissionName][bolo_] = [temp_dict[ch] for ch in self.channel_order["channel_list"][ii]]

        # --- Grab the radDist info and store it
        locdependence = self.fits[evalTime][bestFitID]["info"]["radDists"]["locationdependence"]
        radDistNumber = self.fits[evalTime][bestFitID]["info"]["radDists"]["radDistNumber"]
        loc_ = self.fits[evalTime][bestFitID]["info"]["radDists"]["location"]

        if loc_ is not None:
            self.bestFits[evalTime]["radDistInfo"] = self.data["synthetic"][locdependence][loc_][radDistNumber].info
        else:
            self.bestFits[evalTime]["radDistInfo"] = self.data["synthetic"][locdependence][radDistNumber].info

        # --- Rebuild the radDist
        self.bestFits[evalTime]["radDist"] = self._rebuild_radDist(
            evalTime=evalTime,
            bestFit=True,
        )

    def _post_process_bolo_locations(self, evalTime: float) -> None:
        """Extracts each bolometer phi location from the radDist, stores it under self.info['bolometer_phis']"""

        # --- Only run if self._post_process_fit_arrangement() has been run
        if not hasattr(self, "bestFits"):
            logger.info("Please run self._post_process_fit_arrangement() before " "self._post_process_calculations()")
            return

        # --- Only run if the radDist has been created
        if not "radDist" in self.bestFits[evalTime]:
            logger.info("Please run self._post_process_fit_arrangement() before " "self._post_process_calculations()")
            return

        # --- Extract bolometer phi locations
        bolo_phi = []
        chan_list = {chan for sublist in self.channel_order["channel_list"] for chan in sublist}
        for bolo_ in self.bestFits[evalTime]["radDist"].tokamak.bolometers:
            # A camera is used if any of its channels is in the fit. (This used to
            # compare the camera NAME with channel names, which never matched, so
            # the list was always empty.)
            if chan_list & set(bolo_.info.get("CHANNEL_TAGS", [])):
                bolo_phi.append(bolo_.info["CAMERA_POSITION_R_Z_PHI"][2])

        if not self.info:
            self.info = {}

        # Remove duplications
        bolo_phi_unique = list(set(bolo_phi))

        # Convert to radians
        bolo_phi_unique = np.deg2rad(np.array(bolo_phi_unique))
        self.bestFits[evalTime]["bolometer_phi_locations"] = bolo_phi_unique

    def _post_process_calculations(self, evalTime: float) -> None:
        """
        Toroidal radiated-power profile and peaking factor (TPF) of the best fit.

        Uses uncertainty.toroidal_profile, the same function that produces the
        uncertainty ranges, so the best-fit numbers and the ranges agree. The
        profile is evaluated at the fitted peak location itself (as in the fit's
        residual), no grid sample is removed, and the toroidal mean is the mean
        over the uniform periodic grid (review issue 5).
        """

        if not hasattr(self, "bestFits") or evalTime not in self.bestFits:
            raise RuntimeError("Run _post_process_fit_arrangement() before _post_process_calculations()")

        best = self.bestFits[evalTime]
        params = best["fit"].params.valuesdict()
        synth = best["synthetic_dict"]
        scale_def = self.info["scale_def"]
        bolo_phi_locs = best.get("bolometer_phi_locations")

        prof = uncertainty.toroidal_profile(params, synth, scale_def, bolo_phi_locs)
        quantities, _ = uncertainty.derived_quantities(
            params,
            synth,
            scale_def,
            bolo_phi_locs,
            window_deg=uncertainty.UncertaintyConfig.from_info(self.info).window_deg,
        )

        rad_distribution = {
            "phi": prof["phi"],
            "total_power": prof["total_power"],
            "peak_emission": prof["mu"],
            "toroidal_peaking_factor": quantities["tpf"],
            "quantities": quantities,
        }
        for name, comp in prof["per_emission"].items():
            rad_distribution[name] = {
                "phi": prof["phi"],
                "multiplication_factor": comp["multiplication_factor"],
                "total_power": comp["power"],
            }
        best["radiation_distribution"] = rad_distribution

    def _post_process_uncertainty(self, evalTime: float) -> None:
        """
        Ranges for TPF, total power, peak location and the other quantities in
        uncertainty.QUANTITIES, from parameter covariance, near-equivalent
        radDists and (optionally) camera calibration. Must run before
        _cleanup_fits, which deletes the other candidate fits.

        Stores bestFits[evalTime]['ranges'] (see UncertaintyResult.as_dict).
        """
        cfg = uncertainty.UncertaintyConfig.from_info(self.info)
        if not cfg.enabled:
            return

        fits_t = self.fits[evalTime]
        scores = np.asarray(fits_t["scores"], dtype=float)
        order = [int(i) for i in np.argsort(scores) if np.isfinite(scores[i]) and "fit" in fits_t.get(int(i), {})]
        if not order:
            raise RuntimeError(f"No successful fits at t={evalTime}; cannot compute ranges")

        candidates = [(i, fits_t[i]["fit"], fits_t[i]["synthetic_dict"]) for i in order[: cfg.max_candidates]]

        result = uncertainty.propagate(
            candidates,
            self.fitData[evalTime],
            self.info["scale_def"],
            self.channel_order["bolometer_order"],
            cfg,
            bolo_phi_locs=self.bestFits[evalTime].get("bolometer_phi_locations"),
            helical_endpoint_weight=self.info["helical_endpoint_weight"],
        )
        self.bestFits[evalTime]["ranges"] = result.as_dict()
        logger.info("Ranges at t=%s (total):\n%s", evalTime, uncertainty.format_summary(result))
        for cid, fl in result.flags.items():
            logger.warning("Fit %s: %s", cid, "; ".join(fl))

    # ------------------------------------------------------------------
    # Cleanup / persistence
    # ------------------------------------------------------------------

    def _cleanup_fits(self, evalTime: float) -> None:
        """Delete non-best-fit entries from self.fits to reclaim memory."""

        if self.verbose:
            logger.debug(f"→ Deleting bad fits for = {evalTime:.4f} ms")

        # --- Delete the bad fits to save memory
        if hasattr(self, "fits") and evalTime in self.fits:
            best_id = self.bestFits[evalTime]["bestFitID"]
            for ii in list(self.fits[evalTime].keys()):
                if isinstance(ii, int) and ii != best_id:
                    del self.fits[evalTime][ii]

    def _workspace(self) -> Workspace:
        """The resolved workspace; set when a run config is loaded."""
        if self.ws is None:
            raise RuntimeError("No workspace yet: load a run config first")
        return self.ws

    def _results_dir(self) -> Path:
        """
        Folder for this run's results: a new folder per run
        ({run_dir}/results/{run_id} by default) holding a copy of the run
        config and provenance.yaml.
        """
        if self._results_path is not None:
            return self._results_path
        if self.ws is None or not self.info:
            raise RuntimeError("Load a run config before saving results")

        fields: dict[str, Any] = {
            "tokamak": self.info.get("tokamakName"),
            "shot": self.info.get("shot"),
        }
        if self.run_config_path is not None:
            fields["run"] = self.run_config_path.stem
            fields["run_dir"] = str(self.run_config_path.parent)
        out = self.ws.results_dir(**fields)

        prov = self.ws.provenance(run_config=self.run_config_path, **fields)
        with open(out / "provenance.yaml", "w") as fh:
            yaml.safe_dump(prov, fh, sort_keys=False)
        if self.run_config_path is not None:
            shutil.copy2(self.run_config_path, out / "run_config.yaml")

        self._results_path = out
        logger.info("Results folder: %s", out)
        return out

    def _save_bestFits(self) -> None:
        """Serialise bestFits and fitData to a dill file."""

        def save_results(filename, data):
            with open(filename, "wb") as f:
                dill.dump(data, f)

        if not self.info or "shot" not in self.info or "tokamakName" not in self.info:
            return

        keys = list(self.bestFits.keys())
        if len(keys) > 1:
            t_min = np.min(keys)
            t_max = np.max(keys)
            filename = f"{self.info['shot']}_bestFits_{t_min:.4f}_to_{t_max:.4f}.dill"
        else:
            filename = f"{self.info['shot']}_bestFits_{keys[0]:.4f}.dill"

        pathFileName = self._results_dir() / filename

        save_results(pathFileName, {"fit_data": self.fitData, "bestFits": self.bestFits})
        logger.debug(f"→ Best fits and fitData saved to: {pathFileName}")

    def _load_bestFits(self, path: str = "") -> None:
        """Load bestFits and fitData from a previously saved dill file."""

        def load_results(filename):
            with open(filename, "rb") as f:
                return dill.load(f)

        self.bestFits = {}
        self.fitData = {}

        temp = load_results(path)
        if isinstance(temp, dict):
            for key in list(temp.keys()):
                for evalTime in temp[key]:
                    if key == "fit_data":
                        self.fitData[evalTime] = temp[key][evalTime]
                    elif key == "bestFits":
                        self.bestFits[evalTime] = temp[key][evalTime]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_bestFit(self, evalTime: float, save: bool = False) -> None:
        """
        Best-fit figure for one time. One column per camera (chords over the
        best radDist, signals with the fitted model, residuals), and a last
        column with the radDist's cross-section at the injector and the
        toroidal profile with its ranges. Saved as PNG in the run's results.
        """
        logger.debug("→ Plotting the best fit")

        if not self.info:
            return
        if not hasattr(self, "bestFits"):
            logger.info("Please run _post_process_fit_arrangement() and _post_process_calculations() before _plot_bestFit()")
            return

        best = self.bestFits[evalTime]
        tok = Tokamak(
            tokamakName=self.info["tokamakName"],
            mode="Build",
            reflections=False,
            eqFileName=best["radDistInfo"]["eqFileName"],
            loadBolometers=True,
        )

        bolometerGroups = list(self.info["BOLOMETERS"])
        units_label = UNIT_LABELS.get(best["radDist"].info["units"], "[arb]")
        ncol = len(bolometerGroups) + 1

        f = plt.figure(figsize=(5.0 * ncol, 11.5))
        # Two rows of panels with room for titles between them; the lower row
        # of each column is split again into signals and a residual strip
        gs = f.add_gridspec(2, ncol, height_ratios=[3.0, 3.2], hspace=0.32, wspace=0.3)

        for jj, boloGroupName in enumerate(bolometerGroups):
            # --- Chords over the best radDist at the camera's toroidal location
            self._plot_chord_panel(f.add_subplot(gs[0, jj]), tok, boloGroupName, evalTime)
            # --- Signals and the fitted model, residuals underneath
            lower = gs[1, jj].subgridspec(2, 1, height_ratios=[2.3, 0.9], hspace=0.06)
            ax_sig = f.add_subplot(lower[0])
            ax_res = f.add_subplot(lower[1], sharex=ax_sig)
            self._plot_signal_panel(ax_sig, boloGroupName, evalTime, units_label, legend=(jj == 0), resid_ax=ax_res)

        # --- The best radDist's cross-section at the injector, with a colorbar
        ax = f.add_subplot(gs[0, -1])
        tok._plot_first_wall(ax)
        inj = float(best["synthetic_dict"]["injectionLocation"])
        cf = best["radDist"].plotCrossSection(phi=np.deg2rad(inj), ax=ax)
        if cf is not None:
            f.colorbar(cf, ax=ax, label="emissivity [arb.]", shrink=0.85)
        ax.set_title(f"At the injector ({inj:.1f}°)\n{self._raddist_description(evalTime)}", fontsize=10)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("z [m]")

        # --- Toroidal profile with its ranges, below it
        self._plot_tpf_panel(f.add_subplot(gs[1, -1]), evalTime, tok=tok)

        f.suptitle(self._figure_title(evalTime), fontsize=12)
        f.subplots_adjust(left=0.05, right=0.97, top=0.91, bottom=0.06)

        if save:
            if "shot" in self.info and "tokamakName" in self.info:
                filename = f"{self.info['shot']}_{evalTime:.4f}.png"
                img_dir = self._results_dir() / "images"
                os.makedirs(img_dir, exist_ok=True)
                out_path = img_dir / filename
                f.savefig(out_path, dpi=100, format="png")
                logger.debug(f"→ Figure saved to {out_path}")
            # --- Saved figures are closed, so long runs do not accumulate them
            plt.close(f)
        else:
            plt.show()

    def _figure_title(self, evalTime: float) -> str:
        """Shot, time, chosen radDist and how it was fitted, so a saved PNG describes itself."""
        best = self.bestFits[evalTime]
        path = best.get("radDistInfo", {}).get("radDistPath")
        chosen = Path(path).parent.name + "/" + Path(path).stem if path else best.get("radDistInfo", {}).get("distType", "")
        score = Util_emis3D.fit_score(best["fit"]) if "fit" in best else np.nan
        return (
            f"{self.info['tokamakName']} {self.info['shot']}   t = {evalTime:g}   best radDist: {chosen}\n"
            f"scale_def: {self.info['scale_def']}   errors: {self.info['error_definition']}   "
            f"fit: {self.info['fit_method']}   AIC = {score:.1f}"
        )

    def _raddist_description(self, evalTime: float) -> str:
        """The chosen shape's parameters, from its class's SAVE_KEYS, and its start position."""
        best = self.bestFits[evalTime]
        rd = best.get("radDist")
        info = best.get("radDistInfo", {})
        parts = []
        for key, label in getattr(rd, "SAVE_KEYS", ()):
            if info.get(key) is not None:
                v = info[key]
                parts.append(f"{label} {v:g}" if isinstance(v, (int, float)) else f"{label} {v}")
        if "startR" in info and "startZ" in info:
            parts.append(f"at R {float(info['startR']):.2f}, z {float(info['startZ']):.2f} m")
        return ", ".join(parts)

    def _camera_phis(self, evalTime: float, tok=None) -> list:
        """
        (camera, toroidal angle in degrees) for each camera in the fit, from
        the tokamak's camera configs; falls back to the unnamed locations from
        _post_process_bolo_locations.
        """
        tok = tok if tok is not None else getattr(self.bestFits[evalTime].get("radDist"), "tokamak", None)
        used = {ch for group in self.channel_order["channel_list"] for ch in group} if self.channel_order else set()
        out = {}
        if tok is not None and getattr(tok, "bolometers", None):
            for bolo in tok.bolometers:
                if used & set(bolo.info.get("CHANNEL_TAGS", [])):
                    out.setdefault(bolo.group_name, float(bolo.info["CAMERA_POSITION_R_Z_PHI"][2]))
            return sorted(out.items(), key=lambda kv: kv[1])
        locs = self.bestFits[evalTime].get("bolometer_phi_locations")
        return [("", float(np.rad2deg(p))) for p in np.atleast_1d(locs)] if locs is not None else []

    def _plot_chord_panel(self, ax, tok, boloGroupName: str, evalTime: float) -> None:
        """Plots the first wall, one camera's chords and the radDist cross-section
        at that camera's toroidal location
        """

        tok._plot_first_wall(ax)
        n_chords = sum(len(b.info.get("CHANNEL_TAGS", [])) for b in tok.bolometers if b.group_name == boloGroupName)
        tok._plot_bolometers(ax, boloGroupName, label_every=1 if n_chords <= 12 else 5)

        # Loop over each bolometer to find one that has the same group name
        # This is needed to get the phi location
        phi = None
        for bolo_ in tok.bolometers:
            if bolo_.group_name == boloGroupName:
                phi = tok.get_ave_bolometer_tor_loc(boloGroupName=bolo_.group_name)
                if phi is not None:
                    self.bestFits[evalTime]["radDist"].plotCrossSection(phi=np.deg2rad(phi), ax=ax)
                break

        ax.set_title(boloGroupName if phi is None else f"{boloGroupName} (φ = {phi:.1f}°)")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("z [m]")

    def _plot_signal_panel(
        self, ax, boloName: str, evalTime: float, units_label: str, legend: bool, resid_ax=None
    ) -> None:
        """
        One camera's signals, the fitted model and (with resid_ax) the
        residuals (data - model) / sigma underneath. Channels left out of the
        fit (bad, dead or missing: infinite error) are drawn hollow. The title
        gives the camera's chi-square per channel.
        """
        data = np.asarray(self.fitData[evalTime]["boloData"][boloName], dtype=float)
        err = np.asarray(self.fitData[evalTime]["boloData_error"][boloName], dtype=float)
        channels = self._channel_numbers(boloName)
        used = np.isfinite(err) & np.isfinite(data) & (err > 0)

        ax.errorbar(channels[used], data[used], yerr=err[used], marker="s", ms=5, c="black", linestyle="none", label="data")
        if np.any(~used):
            ax.plot(channels[~used], np.nan_to_num(data[~used]), marker="s", ms=6, mfc="none", mec="black",
                    linestyle="none", label="not fitted")

        # --- Fitted emission components (only when there is more than one) and their total
        synth = self.bestFits[evalTime]["synthData"]
        total = np.zeros(data.size)
        for jj, emissionName in enumerate(synth):
            em = np.asarray(synth[emissionName][boloName], dtype=float)
            total += em
            if len(synth) > 1:
                ax.plot(channels, em, marker=PLOT_MARKERS[jj % len(PLOT_MARKERS)], color=PLOT_COLORS[jj % len(PLOT_COLORS)],
                        label=f"{emissionName}")
        ax.plot(channels, total, color="purple", label="model" if len(synth) == 1 else "total", linewidth=2.5)

        # --- y-limit from the fitted channels and the model
        vals = np.concatenate((data[used] + err[used], total)) if np.any(used) else total
        ymax = np.nanmax(vals) if vals.size else np.nan
        if np.isfinite(ymax) and ymax > 0:
            ax.set_ylim(0, float(ymax) * 1.05)

        # --- Fit quality of this camera
        r = (data[used] - total[used]) / err[used]
        title = boloName
        if r.size:
            title += f"   χ²/N = {np.sum(r**2) / r.size:.2f}  ({r.size} channels)"
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(f"Emission {units_label}")
        if legend:
            ax.legend(fontsize=8)

        if resid_ax is None:
            ax.set_xlabel("Channel Number")
            return

        # --- Residual strip: within +-1 sigma is shaded, +-2 sigma dotted
        ax.tick_params(labelbottom=False)
        resid_ax.axhspan(-1, 1, color="0.9", lw=0)
        for y in (-2, 2):
            resid_ax.axhline(y, color="0.6", linestyle=":", lw=0.8)
        resid_ax.axhline(0, color="purple", lw=1.0)
        resid_ax.plot(channels[used], r, marker="o", ms=4, color="black", linestyle="none")
        lim = max(3.0, float(np.nanmax(np.abs(r))) * 1.15) if r.size else 3.0
        resid_ax.set_ylim(-lim, lim)
        resid_ax.set_ylabel("(data − model)/σ", fontsize=8)
        resid_ax.set_xlabel("Channel Number")

    def _plot_tpf_panel(self, ax, evalTime: float, tok=None) -> None:
        """
        Toroidal radiation profile with its 68%/95% ranges, plotted against the
        toroidal angle from the injector (so the peak is in the middle), with
        the cameras' locations marked: the profile is measured there and
        modelled in between. TPF and total power, with ranges, in the title.
        """
        best = self.bestFits[evalTime]
        rad = best["radiation_distribution"]
        ranges = best.get("ranges")
        inj = float(best["synthetic_dict"]["injectionLocation"])

        def from_injector(deg):
            return (np.asarray(deg, dtype=float) - inj + 180.0) % 360.0 - 180.0

        x = from_injector(np.rad2deg(rad["phi"]))
        order = np.argsort(x)
        x = x[order]
        y = np.asarray(rad["total_power"], dtype=float)[order]

        y_max = np.nanmax(y)
        if ranges is not None:
            y_max = max(y_max, np.nanmax(ranges["profile_band"]["95%"][1]))
        scale = np.floor(np.log10(y_max)) if np.isfinite(y_max) and y_max > 0 else 0.0
        norm = 10.0**scale

        if ranges is not None:
            band = ranges["profile_band"]
            for level, colour in (("95%", "0.85"), ("68%", "0.65")):
                lo = np.asarray(band[level][0])[order] / norm
                hi = np.asarray(band[level][1])[order] / norm
                ax.fill_between(x, lo, hi, color=colour, lw=0, label=f"profile {level} range")
        ax.plot(x, y / norm, color="black", linewidth=2.0, label="best fit")

        # --- Cameras: where the profile is measured (cameras at one angle share a marker)
        at_angle = {}
        for name, phi in self._camera_phis(evalTime, tok=tok):
            at_angle.setdefault(round(phi % 360.0, 3), []).append(name)
        for phi, names in sorted(at_angle.items()):
            name = ", ".join(n for n in names if n)
            xc = float(from_injector(phi))
            ax.axvline(xc, color="tab:blue", linestyle=":", lw=1.2)
            if name:
                # label on the inside of the line near the left edge, so it is not cut off
                ha = "left" if xc < -170.0 else "right"
                ax.text(xc, 0.98, f" {name} ", rotation=90, ha=ha, va="top", fontsize=8, color="tab:blue",
                        transform=ax.get_xaxis_transform())

        # --- Peak: at the injector unless its location was fitted
        x_peak = float(from_injector(np.rad2deg(rad["peak_emission"]) % 360.0))
        ax.axvline(x_peak, linestyle="dashed", color="tab:red")
        ax.text(x_peak, 0.02, " peak", ha="left", va="bottom", fontsize=9, color="tab:red",
                transform=ax.get_xaxis_transform())

        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_ylim(bottom=0)
        ax.set_xlabel(f"toroidal angle from the injector at {inj:.1f}° [deg]")
        ax.set_ylabel(f"radiated power per radian [$10^{{{int(scale)}}}$ arb.]")

        title = f"TPF {rad['toroidal_peaking_factor']:.2f}"
        P_tot = rad.get("quantities", {}).get("total_power")
        if ranges is not None and "tpf" in ranges["total"]:
            lo, hi = ranges["total"]["tpf"]["68%"]
            title += f" (68%: {lo:.2f}–{hi:.2f})"
        if P_tot is not None:
            title += f"\ntotal power {P_tot:.3g}"
            if ranges is not None and "total_power" in ranges["total"]:
                lo, hi = ranges["total"]["total_power"]["68%"]
                title += f" (68%: {lo:.3g}–{hi:.3g})"
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="lower center", framealpha=0.9)

    def _channel_numbers(self, boloName: str) -> np.ndarray:
        """
        True channel numbers for a bolometer, taken from the master channel
        order (self.channel_order) so that channels missing from the data do
        not shift the numbering. Falls back to 1..N positions for names
        without trailing digits.
        """
        chan_names = None
        if self.channel_order is not None:
            for ii, name_ in enumerate(self.channel_order["bolometer_order"]):
                if name_ == boloName:
                    chan_names = self.channel_order["channel_list"][ii]
                    break

        if chan_names is None:
            chan_names = list(self.data["observed"][boloName]["channelOrder"])

        numbers = []
        for jj, chan in enumerate(chan_names):
            num_ = extract_end_numbers(str(chan))
            numbers.append(int(num_) if num_ is not None else jj + 1)

        return np.array(numbers)

    # ------------------------------------------------------------------
    # Placeholder / future work
    # ------------------------------------------------------------------

    def _combine_synthetics_for_fits(self, evalTime: float) -> None:
        """
        Combines radDists from up to two injection locations iteravely

        """

        pass

        """
        Do something like this:

        def combine_dicts(a, b=None):
            c = {}
            counter = 0

            if b is None:
                # Only loop over a
                for a_key in a.keys():
                    c[counter] = {
                        "index a": a_key,
                        "data a": a[a_key]["data"]
                    }
                    counter += 1
            else:
                # Loop over all permutations of a and b
                for a_key, b_key in itertools.product(a.keys(), b.keys()):
                    c[counter] = {
                        "index a": a_key,
                        "index b": b_key,
                        "data a": a[a_key]["data"],
                        "data b": b[b_key]["data"]
                    }
                    counter += 1
            return c
        """
