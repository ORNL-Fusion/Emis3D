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
4. crossCalib Flag set to True should only be used to find cross-calibration factors between bolometers for a specific shot.
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
import logging
import time
import warnings

import dill
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, report_fit
from scipy.integrate import simpson

import main.Util_emis3D as Util_emis3D

logger = logging.getLogger(__name__)
from main.Globals import EMIS3D_INPUTS_DIRECTORY
from main.Tokamak import Tokamak
from main.Util import (
    config_loader,
    convert_arrays_to_list,
    extract_end_numbers,
    find_max_nested_lists,
    get_filenames_in_directory,
    read_h5,
)
from main.radDistFitting import RadDistFitting
from main.radDist import Helical, ElongatedRing, HelicalRing

# --- Fit weight assigned to dead / zero-signal channels so the minmizer ignores
# them.
DEAD_CHANNEL_ERROR = 1.0e8

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
    ):
        """
        Main class for running the emis3D program.

        Parameters
        ----------
        tokamakName   : Name of the tokamak
        runConfigName : Name of the run configuration file
        initialize    : When True the standard initialisation pipeline is run
        verbose       : Print extra progress information
        """

        # --- Initialize variables
        self.data = {}
        self.info = None
        self.verbose = verbose

        # Configure logging level for the whole 'main' package
        if verbose:
            logging.getLogger("main").setLevel(logging.DEBUG)

        if initialize:
            self._initialize(tokamakName=tokamakName, runConfigName=runConfigName)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(
        self, tokamakName: str | None = None, runConfigName: str | None = None
    ) -> None:
        """Load configuration, bolometer data and radDists."""
        if tokamakName is None or runConfigName is None:
            raise ValueError(
                "tokamakName and runConfigName are required to initialise Emis3D"
            )

        Util_emis3D.print_intro()
        self._load_config_file(tokamakName=tokamakName, runConfigName=runConfigName)
        self._load_bolometer_data()
        self._create_master_channel_order()
        self._load_radDists()
        self._prune_channel_order_to_radDists()

    def _perform_fits(self, evalTime: float, crossCalib: bool = False) -> None:
        """
        Prepares the data, preforms the fits

        Parameters
        ----------
        evalTime   : Time to run the fit
        crossCalib : Flag to tell the program to find the calibration
                     factor between bolometers during the CQ
        """

        # Added a bunch of timing flags, to see where improvement can be made
        self._prepare_fits(evalTime=evalTime, crossCalib=crossCalib)
        t_start = time.time()

        self._minimize_radDists(evalTime=evalTime, crossCalib=crossCalib)
        logger.info(f"→ Fitting done in {time.time() - t_start:.2f} seconds")
        t_start = time.time()
        self._post_process_fit_arrangement(evalTime=evalTime, crossCalib=crossCalib)

        logger.info(
            f"→ Done with fit post-processing step 1 out of 2 in {time.time() - t_start:.2f} seconds"
        )
        t_start = time.time()

        self._post_process_bolo_locations(evalTime=evalTime)
        self._post_process_calculations(evalTime=evalTime)
        logger.info(
            f"→ Done with fit post-processing step 2 out of 2 in {time.time() - t_start:.2f} seconds"
        )

    # ------------------------------------------------------------------
    # Configuration / data loading
    # ------------------------------------------------------------------

    def _load_config_file(
        self,
        tokamakName: str | None = None,
        runConfigName: str | None = None,
        pathFileName: str | None = None,
    ) -> None:
        """
        Loads the YAML configuration file for the given tokamak

        Either supply ``pathFileName`` directly, or supply both
        ``tokamakName`` and ``runConfigName`` to have the path constructed
        automatically.
        """

        if pathFileName is None:
            if tokamakName is not None and runConfigName is not None:
                pathFileName = str(
                    EMIS3D_INPUTS_DIRECTORY / tokamakName / "runs" / runConfigName
                )
            else:
                raise ValueError(
                    "tokamakName, runConfigName, or pathFileName is None; "
                    "cannot load config file"
                )

        if not os.path.isfile(str(pathFileName)):
            raise FileNotFoundError(f"Config file not found: {pathFileName}")

        self.info = config_loader(str(pathFileName), verbose=self.verbose)

        if self.info is None:
            raise RuntimeError(f"Could not load the configuration file: {pathFileName}")

        self.info["tokamakName"] = tokamakName
        self.info["runConfigName"] = runConfigName

    def _load_radDists(self) -> None:
        """
        Loads each radDist within the directories listed in the config file:
        radDistDirectories_LocIndependent, and radDistDirectories_LocDependent
        """

        if self.verbose:
            logger.debug("Loading radDists")

        # --- Initialise synthetic signal arrays
        self.data.update({"synthetic": {"locDependent": {}, "locIndependent": {}}})
        self.files = []

        if self.info is None or (
            "radDistDirectories_LocIndependent" not in self.info
            and "radDistDirectories_LocDependent" not in self.info
        ):
            raise RuntimeError("No radDistDirectories found in the config file")

        # --- Load location independent radDists
        count_ = 0
        for dir_ in self.info["radDistDirectories_LocIndependent"]:
            pathFileName = os.path.join(
                EMIS3D_INPUTS_DIRECTORY,
                self.info["tokamakName"],
                "radDists",
                dir_,
            )
            for file_ in get_filenames_in_directory(pathFileName):
                try:
                    temp_ = RadDistFitting(radDistPath=file_)
                    self.data["synthetic"]["locIndependent"][count_] = temp_
                    count_ += 1
                except Exception as e:
                    warnings.warn(f"Skipping radDist {file_}: {e}", stacklevel=2)
                    logger.warning("Skipping radDist %s: %s", file_, e)

        # --- Load location dependent radDists
        for ii, dir_ in enumerate(self.info["radDistDirectories_LocDependent"]):
            self.data["synthetic"]["locDependent"][f"loc_{ii}"] = {}
            pathFileName = os.path.join(
                EMIS3D_INPUTS_DIRECTORY,
                self.info["tokamakName"],
                "radDists",
                dir_,
            )
            count_ = 0
            for file_ in get_filenames_in_directory(pathFileName):
                try:
                    temp_ = RadDistFitting(radDistPath=file_)
                    self.data["synthetic"]["locDependent"][f"loc_{ii}"][count_] = temp_
                    count_ += 1
                except Exception as e:
                    warnings.warn(f"Skipping radDist {file_}: {e}", stacklevel=2)
                    logger.warning("Skipping radDist %s: %s", file_, e)

        if self.verbose:
            logger.debug("Done loading radDists")

    def _load_bolometer_data(self) -> None:
        """
        Load pre-calibrated SXR/bolometer data from ``inputs/{tokamakName}/sxrData/``.
        The filename is taken from ``dataFileName`` in the run config.
        """

        if self.info is None or "BOLOMETERS" not in self.info:
            raise RuntimeError("No BOLOMETERS found in the config file")

        self.data["observed"] = {}
        for bolo_ in self.info["BOLOMETERS"]:
            pathFileName = os.path.join(
                EMIS3D_INPUTS_DIRECTORY,
                self.info["tokamakName"],
                "sxrData",
                self.info["BOLOMETERS"][bolo_]["dataFileName"],
            )
            if not os.path.isfile(pathFileName):
                raise FileNotFoundError(
                    f"Bolometer data file not found: {pathFileName}"
                )

            logger.debug("Loading bolometer data: %s", pathFileName)

            temp_ = read_h5(pathFileName)

            for ii, ch_ in enumerate(temp_["channelOrder"]):
                temp_["channelOrder"][ii] = ch_.decode("utf-8")

            temp_["DATA_CALIBRATED"] *= self.info["BOLOMETERS"][bolo_].get(
                "scalingFactor", 1.0
            )

            self.data["observed"][bolo_] = temp_

    def _create_master_channel_order(self) -> None:
        """
        Build a master channel list from the loaded bolometer data.
        All radDists and raw data will be ordered to match this list.

        List is of the format:
        [[bolo1_1, bolo1_2, ...], [bolo2_1, bolo2_2, ...], ...]
        """

        if self.info is None or "BOLOMETERS" not in self.info:
            raise RuntimeError("No BOLOMETERS found in the config file")

        self.channel_order = {
            "bolometer_order": [],
            "channel_list": [],
        }

        for bolo in self.data["observed"]:
            self.channel_order["bolometer_order"].append(bolo)
            self.channel_order["channel_list"].append(
                list(self.data["observed"][bolo]["channelOrder"])
            )

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
                "Dropped %d channel(s) absent from the radDist synthetic data; "
                "they will not be passed to the minimizer: %s",
                len(dropped),
                ", ".join(str(ch) for ch in dropped),
            )

    # ------------------------------------------------------------------
    # Fit preparation
    # ------------------------------------------------------------------

    def _average_observed_data(
        self, arrayName: str = "", evalTime: float | None = None
    ) -> list:
        """
        Average SXR/bolometer data over a window [evalTime-dt, evalTime+dt].

        Returns a flat list of averaged channel values, or zeros if evalTime
        is None or dt is not configured.
        """

        if evalTime is None:
            return [0] * self.data["observed"][arrayName]["NUM_CHANNELS"]

        time_ = self.data["observed"][arrayName]["TIME"]

        if self.info is None or "dt" not in self.info:
            return [1] * self.data["observed"][arrayName]["NUM_CHANNELS"]

        dt_ = self.info["dt"]
        start = np.abs(time_ - (evalTime - dt_)).argmin()
        end = np.abs(time_ - (evalTime + dt_)).argmin()

        # --- Average the data, include endpoint to prevent errors when dt is too small
        vals = np.mean(
            self.data["observed"][arrayName]["DATA_CALIBRATED"][:, start : end + 1],
            axis=1,
        )

        # --- Replace NaNs with zero
        vals = np.where(np.isnan(vals), 0.0, vals)

        return convert_arrays_to_list(vals)

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
        data_ = {}
        for bolo_ in self.data["observed"]:
            temp = self._average_observed_data(arrayName=bolo_, evalTime=evalTime)
            data_.update(dict(zip(self.data["observed"][bolo_]["channelOrder"], temp)))

        self.fitData[evalTime]["dataMap"] = data_

        # --- Find the max value in all the channels
        max_ = max(
            (data_[ch] for bolo_ in self.channel_order["channel_list"] for ch in bolo_),
            default=0.0,
        )

        for ii, bolo_chans in enumerate(self.channel_order["channel_list"]):
            temp = []
            temp_e = []

            # --- Find the max value for that specific array
            """
            max_ = 0
            for channel in channels:
                if data_[channel] > max_:
                    max_ = data_[channel]
            """

            for channel in bolo_chans:
                val = data_[channel]
                temp.append(val)

                # --- Large error for zero signal, use definition otherwise
                if val > 1.0:
                    err_frac = 0.05
                    if self.info is not None:
                        err_frac = Util_emis3D.signal_error(
                            self.info["error_definition"], val, max_, scale_factor=1.0
                        )
                    err_ = val * err_frac
                else:
                    err_ = np.float64(DEAD_CHANNEL_ERROR)

                temp_e.append(err_)

            self.fitData[evalTime]["observed"].append(temp)
            self.fitData[evalTime]["observed_error"].append(temp_e)

            # --- Temp is a nested list of bolometers [[Bolo1-1, Bolo1-2, ...], [Bolo2-1, Bolo2-2], etc.]
            bolo_name = self.channel_order["bolometer_order"][ii]
            self.fitData[evalTime]["boloData"][bolo_name] = temp
            self.fitData[evalTime]["boloData_error"][bolo_name] = temp_e

        if self.verbose:
            logger.debug("→ Observed data prepared for fitting")

    def _prepare_synthetic_for_fits(
        self, evalTime: float, crossCalib: bool = False
    ) -> None:
        """Prepare and parameterise every radDist for the minimization."""

        print_minor_error = True

        def print_error(radD):
            logger.info("")
            logger.warning("-" * 10 + "MINOR ERROR" + "-" * 10)
            logger.warning(
                f"Channels: {radD.info['ERROR CHANNELS']}were not found in the radDist, they will be ignored"
            )
            logger.warning("-" * 31 + "\n")
            logger.info("")

        logger.debug("→ Preparing synthetic data for fitting")

        # --- Scale the synthetic data to observed for better fitting
        max_data_val = find_max_nested_lists(self.fitData[evalTime]["observed"])

        if self.info is not None and "enable_dphi_scaling" in self.info:

            boloNames = self.channel_order["bolometer_order"] if crossCalib else None

            # --- Arrange and create parameters for the location dependent data
            for loc in self.data["synthetic"]["locDependent"]:
                for number_ in self.data["synthetic"]["locDependent"][loc]:
                    radD = self.data["synthetic"]["locDependent"][loc][number_]
                    radD.prepare_for_fits(
                        self.channel_order["channel_list"],
                        data_max=max_data_val,
                    )

                    radD.create_parameters(
                        boloNames=boloNames,
                        enable_dphi_scaling=self.info["enable_dphi_scaling"],
                        vary_peak_rad_location=self.info["vary_peak_rad_location"],
                        scale_def=self.info["scale_def"],
                    )
                    if "ERROR CHANNELS" in radD.info and print_minor_error:
                        print_error(radD)
                        print_minor_error = False

            # --- Arrange and create parameters for the location independent data
            for number_ in self.data["synthetic"]["locIndependent"]:
                radD = self.data["synthetic"]["locIndependent"][number_]
                radD.prepare_for_fits(
                    self.channel_order["channel_list"], data_max=max_data_val
                )
                radD.create_parameters(
                    boloNames=boloNames,
                    enable_dphi_scaling=self.info["enable_dphi_scaling"],
                    vary_peak_rad_location=self.info["vary_peak_rad_location"],
                    scale_def=self.info["scale_def"],
                )

                if "ERROR CHANNELS" in radD.info and print_minor_error:
                    print_error(radD)
                    print_minor_error = False

        if self.verbose:
            logger.debug("→ Done preparing synthetic data for fit")

    def _build_synthetic_dict(
        self, radDist_, locDependence: str, number, location=None
    ) -> dict:
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
                "observed_phi_loc": radDist_.fitSynthetic[emissionName][
                    "observed_phi_loc"
                ],
                "data": radDist_.fitSynthetic[emissionName]["data"],
                "data_error": radDist_.fitSynthetic[emissionName]["data_error"],
            }

            # --- Toroidal radiated-power profile, used by the helical endpoint
            # continuity constraint (Util_emis3D.helical_endpoint_penalty).
            # Stored under 'phi_array'/'P_pol' on the radDist; absent for older
            # saved radDists, in which case the constraint quietly turns off.
            tor_power = radDist_.data.get("toroidalRadiatedPower", {})
            if emissionName in tor_power:
                synthetic_dict[emissionName]["phi_array"] = tor_power[emissionName].get(
                    "phi_array"
                )
                synthetic_dict[emissionName]["P_pol"] = tor_power[emissionName].get(
                    "P_pol"
                )

        return synthetic_dict

    def _prepare_fits(self, evalTime: float, crossCalib: bool = False) -> None:
        """
        Organise data and synthetic signals into self.fits[evalTime] in
        preparation for _minimize_radDists.
        """

        if self.verbose:
            logger.debug("→ Preparing data for fitting")

        self._prepare_data_for_fit(evalTime=evalTime)
        self._prepare_synthetic_for_fits(evalTime=evalTime, crossCalib=crossCalib)

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
                "synthetic_dict": self._build_synthetic_dict(
                    radDist_, "locIndependent", number
                ),
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
                    "synthetic_dict": self._build_synthetic_dict(
                        radDist_, "locDependent", number, location=loc_
                    ),
                }

        num_fits = len(self.fits[evalTime])
        if self.info is not None:
            self.info["numFits"] = num_fits
        self.fits[evalTime]["chiSqVec"] = np.full(num_fits, 1.0e19)

    # ------------------------------------------------------------------
    # Minimization
    # ------------------------------------------------------------------

    def _minimize_radDists(self, evalTime: float, crossCalib: bool = False) -> None:
        """
        Run leastsq minimization for every fit candidate stored in
        self.fits[evalTime].
        """

        try:
            if self.info is None or "scale_def" not in self.info:
                return

            # --- Data used for fitting
            data_dict = self.fitData[evalTime]

            # --- Soft-constraint weight tying helical endpoints together.
            # Active (1.0) by default; set to 0.0 in the run config to disable.
            helical_endpoint_weight = self.info.get("helical_endpoint_weight", 1.0)

            for ii in self.fits[evalTime]:

                if not isinstance(ii, int):
                    continue

                if ii % 1_000 == 0 and self.verbose:
                    logger.debug(f"→ Preforming fit {ii} out of {self.info['numFits']}")

                synth_dict = self.fits[evalTime][ii]["synthetic_dict"]
                pars = self.fits[evalTime][ii]["parameters"]

                try:
                    boloNames = None
                    # --- Include bolometer names if doing a cross-calibration
                    if crossCalib and (
                        self.channel_order is not None
                        and "bolometer_order" in self.channel_order
                    ):
                        boloNames = self.channel_order["bolometer_order"]

                    self.fits[evalTime][ii]["fit"] = minimize(
                        Util_emis3D.residual,
                        pars,
                        args=(
                            data_dict,
                            synth_dict,
                            self.info["scale_def"],
                            boloNames,
                            True,  # residual = True
                            helical_endpoint_weight,
                        ),
                        method="differential_evolution",
                    )
                    """
                    results_global = minimize(
                        Util_emis3D.residual,
                        pars,
                        args=(
                            data_dict,
                            synth_dict,
                            self.info["scale_def"],
                            boloNames,
                            True,  # residual = True
                            helical_endpoint_weight,
                        ),
                        method = "differential_evolution",
                    )

                    self.fits[evalTime][ii]["fit"] = minimize(
                        Util_emis3D.residual,
                        results_global.params, # type:ignore
                        args=(
                            data_dict,
                            synth_dict,
                            self.info["scale_def"],
                            boloNames,
                            True,  # residual = True
                            helical_endpoint_weight,
                        ),
                        method="least_squares",
                    )
                    """

                    self.fits[evalTime]["chiSqVec"][ii] = self.fits[evalTime][ii][
                        "fit"
                    ].redchi
                except Exception as e:
                    logger.warning(f"An error occurred during the {ii} iteration: {e}")
                    self.fits[evalTime]["chiSqVec"][ii] = 1.0e19

        except Exception as e:
            logger.warning(f"An error occurred while fitting: {e}")

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _bestFit_radDist_parameters(
        self, evalTime: float, emissionName: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Finds the starting and ending phi for a given radDist"""
        phi = self.bestFits[evalTime]["radDist"].data["toroidalRadiatedPower"][
            emissionName
        ]["phi_array"]
        P_pol = self.bestFits[evalTime]["radDist"].data["toroidalRadiatedPower"][
            emissionName
        ]["P_pol"]
        return np.array(phi), np.array(P_pol)

    def _bestFit_radDist_scaling(self, evalTime: float, emissionName: str) -> float:
        """Returns the pre-fit synthetic / data normalization factor"""
        norm_factor = self.bestFits[evalTime]["synthetic_dict"][emissionName][
            "scaleSynth"
        ]
        return norm_factor

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
            radDist_ = Helical(
                config=rad_.info, startR=rad_.info["startR"], startZ=rad_.info["startZ"]
            )

        elif info["distType"].lower() == "elongatedring":
            radDist_ = ElongatedRing(config=rad_.info)
        elif info["distType"].lower() == "helicalring":
            radDist_ = HelicalRing(
                config=rad_.info, startR=rad_.info["startR"], startZ=rad_.info["startZ"]
            )
        else:
            logger.info(
                f"_rebuild_radDist() only supports Helical, ElongatedRing, or HelicalRing"
            )
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

    def _post_process_fit_arrangement(
        self, evalTime: float, crossCalib: bool = False
    ) -> None:
        """Identify the best fit and reorganise synthetic data by bolometer."""

        # --- Find the best fit
        bestFitID = np.array(self.fits[evalTime]["chiSqVec"]).argmin().item()

        # --- Print the results of the best fit
        print("\n-----------Best Fit-----------")
        report_fit(self.fits[evalTime][bestFitID]["fit"])
        print("------------------------------\n")

        # --- Store the best fit
        if not hasattr(self, "bestFits"):
            self.bestFits = {}

        self.bestFits[evalTime] = self.fits[evalTime][bestFitID]
        self.bestFits[evalTime]["bestFitID"] = bestFitID

        if self.info is None or "scale_def" not in self.info:
            logger.info("No scale_def found in the config file")
            return

        boloNames = None
        if crossCalib and (
            self.channel_order is not None and "bolometer_order" in self.channel_order
        ):
            boloNames = self.channel_order["bolometer_order"]

        # --- First multiply the synthetic data by the fit parameters
        data_ = Util_emis3D.residual(
            self.bestFits[evalTime]["fit"].params,
            None,
            self.bestFits[evalTime]["synthetic_dict"],
            self.info["scale_def"],
            boloNames=boloNames,
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
                self.bestFits[evalTime]["synthData"][emissionName][bolo_] = [
                    temp_dict[ch] for ch in self.channel_order["channel_list"][ii]
                ]

        # --- Grab the radDist info and store it
        locdependence = self.fits[evalTime][bestFitID]["info"]["radDists"][
            "locationdependence"
        ]
        radDistNumber = self.fits[evalTime][bestFitID]["info"]["radDists"][
            "radDistNumber"
        ]
        loc_ = self.fits[evalTime][bestFitID]["info"]["radDists"]["location"]

        if loc_ is not None:
            self.bestFits[evalTime]["radDistInfo"] = self.data["synthetic"][
                locdependence
            ][loc_][radDistNumber].info
        else:
            self.bestFits[evalTime]["radDistInfo"] = self.data["synthetic"][
                locdependence
            ][radDistNumber].info

        # --- Rebuild the radDist
        self.bestFits[evalTime]["radDist"] = self._rebuild_radDist(
            evalTime=evalTime,
            bestFit=True,
        )

    def _post_process_bolo_locations(self, evalTime: float) -> None:
        """Extracts each bolometer phi location from the radDist, stores it under self.info['bolometer_phis']"""

        # --- Only run if self._post_process_fit_arrangement() has been run
        if not hasattr(self, "bestFits"):
            logger.info(
                "Please run self._post_process_fit_arrangement() before "
                "self._post_process_calculations()"
            )
            return

        # --- Only run if the radDist has been created
        if not "radDist" in self.bestFits[evalTime]:
            logger.info(
                "Please run self._post_process_fit_arrangement() before "
                "self._post_process_calculations()"
            )
            return

        # --- Extract bolometer phi locations
        bolo_phi = []
        chan_list = [
            chan for sublist in self.channel_order["channel_list"] for chan in sublist
        ]
        for bolo_ in self.bestFits[evalTime]["radDist"].tokamak.bolometers:
            # First check to see if the name is in the master channel order
            bolo_name = bolo_.info["NAME"]
            if bolo_name in chan_list:
                bolo_phi.append(bolo_.info["CAMERA_POSITION_R_Z_PHI"][2])

        if self.info is None:
            self.info = {}

        # Remove duplications
        bolo_phi_unique = list(set(bolo_phi))

        # Convert to radians
        bolo_phi_unique = np.deg2rad(np.array(bolo_phi_unique))
        self.bestFits[evalTime]["bolometer_phi_locations"] = bolo_phi_unique

    def _post_process_calculations(self, evalTime: float) -> None:
        """
        Rebuild radDist, calculate powerPerBin, and compute the toroidal
        peaking factor (TPF).
        """

        # --- Only run if self._post_process_fit_arrangement() has been run
        if not hasattr(self, "bestFits"):
            logger.info(
                "Please run self._post_process_fit_arrangement() before "
                "self._post_process_calculations()"
            )
            return

        if self.info is None:
            raise AttributeError("Self.info should not be none!")

        # --- Create an empty dictionary
        self.bestFits[evalTime]["radiation_distribution"] = {}
        rad_distribution = self.bestFits[evalTime]["radiation_distribution"]
        emissionNames = self.bestFits[evalTime]["synthetic_dict"]["emissionNames"]

        scale_def = self.info["scale_def"]
        params = self.bestFits[evalTime]["fit"].params.valuesdict()

        # --- Check to see if the peak radiation location can be varied
        mu = self.bestFits[evalTime]["synthetic_dict"]["injectionLocation_rad"]
        if "peak_rad_loc" in params:
            mu = float(params["peak_rad_loc"])

        # Value at the end of each tag in params
        inj_loc_tag = Util_emis3D.loc_tag(
            self.bestFits[evalTime]["synthetic_dict"]["injectionLocation"]
        )
        mu_grid = None

        for emissionName in emissionNames:
            rad_distribution[emissionName] = {}

            # Find the phi limits that the radDist was created with
            rD_phi, rD_P_pol = self._bestFit_radDist_parameters(evalTime, emissionName)
            rD_scale = self._bestFit_radDist_scaling(evalTime, emissionName)

            # Create total power arrays
            if "phi" not in rad_distribution:
                rad_distribution["phi"] = rD_phi
                rad_distribution["total_power"] = np.zeros(rD_phi.shape)
            else:
                # Checker to make sure that phi arrays match when doing helical distributions
                # (or more than one injection location). Use array_equal because
                # 'rD_phi != stored' is an element-wise array, which is ambiguous
                # in a boolean context and would raise a ValueError.
                if not np.array_equal(rD_phi, rad_distribution["phi"]):
                    raise ValueError(
                        "Error! PHI arrays do not match in _post_process_calculations"
                    )

            # dphi already accounts for _rev1, _rev2, etc. for helical distributions
            # Snap mu to the phi grid, so mu is zero at the injection location
            ang = (rD_phi - mu + np.pi) % (2.0 * np.pi) - np.pi
            mu_grid = float(rD_phi[int(np.argmin(np.abs(ang)))])
            dphi = Util_emis3D.find_dphi(rD_phi, mu_grid, emissionName=emissionName)

            # Both directions share amplitude 'a'; their individual decay constant
            # 'b' controls how fast each falls off away from the injection location.
            a = params[f"a_{inj_loc_tag}"]
            b = params[f"b_{emissionName}_{inj_loc_tag}"]

            scale_ = Util_emis3D.scale_wrapper(
                a=np.array([a,b]),
                phi=rD_phi,
                dphi=dphi,
                mu=mu_grid,
                scale_def=scale_def,
                emissionName=emissionName,
                bolo_phi_locs=self.bestFits[evalTime]["bolometer_phi_locations"],
            )

            # --- Now calculate the radiation around the vessel due to the radDist
            rD_power = rD_P_pol * scale_ * rD_scale

            rad_distribution[emissionName]["phi"] = np.asarray(rD_phi)
            rad_distribution[emissionName]["multiplication_factor"] = np.asarray(scale_)
            rad_distribution[emissionName]["total_power"] = np.asarray(rD_power)
            rad_distribution["total_power"] += rD_power

        rad_distribution["peak_emission"] = mu

        # --- Remove the spot at the injection location, since it is double-counted
        if mu_grid is not None:
            loc = np.abs(rad_distribution["phi"] - mu_grid).argmin()
            rad_distribution["total_power"] = np.delete(
                rad_distribution["total_power"], loc
            )
            rad_distribution["phi"] = np.delete(rad_distribution["phi"], loc)

        tp = rad_distribution["total_power"]
        tpf = np.max(tp) / (simpson(tp, x=rad_distribution["phi"]) / (2.0 * np.pi))
        rad_distribution["toroidal_peaking_factor"] = tpf

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

    def _save_bestFits(self) -> None:
        """Serialise bestFits and fitData to a dill file."""

        def save_results(filename, data):
            with open(filename, "wb") as f:
                dill.dump(data, f)

        if (
            self.info is None
            or "shot" not in self.info
            or "tokamakName" not in self.info
        ):
            return

        keys = list(self.bestFits.keys())
        if len(keys) > 1:
            t_min = np.min(keys)
            t_max = np.max(keys)
            filename = f"{self.info['shot']}_bestFits_{t_min:.4f}_to_{t_max:.4f}.dill"
        else:
            filename = f"{self.info['shot']}_bestFits_{keys[0]:.4f}.dill"

        pathFileName = (
            EMIS3D_INPUTS_DIRECTORY
            / self.info["tokamakName"]
            / "runs"
            / str(self.info["shot"])
            / filename
        )

        save_results(
            pathFileName, {"fit_data": self.fitData, "bestFits": self.bestFits}
        )
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
        """Plots the fit synthetic signal, data, and radDist for the given evalTime."""
        logger.debug(f"→ Plotting the best fit")

        if self.info is None:
            return

        # --- Rebuild the bestFit radDist
        if not hasattr(self, "bestFits"):
            logger.info(
                "Please run _post_process_fit_arrangement() and "
                "_post_process_calculations() before _plot_bestFit()"
            )
            return

        # --- Find the eqFileName, if it exists
        eqFileName = self.bestFits[evalTime]["radDistInfo"]["eqFileName"]

        tok = Tokamak(
            tokamakName=self.info["tokamakName"],
            mode="Build",
            reflections=False,
            eqFileName=eqFileName,
            loadBolometers=True,
        )

        bolometerGroups = list(self.info["BOLOMETERS"])
        units = self.bestFits[evalTime]["radDist"].info["units"]
        units_label = UNIT_LABELS.get(units, "[arb]")

        num_columns = len(bolometerGroups) + 1
        f = plt.figure(figsize=(15, 8))

        # --- Plot the bolometer chords and radDist contour
        count_ = 0
        # Bolometers is a list
        for boloGroupName in bolometerGroups:
            count_ += 1
            ax = f.add_subplot(2, num_columns, count_)
            self._plot_chord_panel(ax, tok, boloGroupName, evalTime)

        # --- Plot the contour at the injection location
        count_ += 1
        ax = f.add_subplot(2, num_columns, count_)
        tok._plot_first_wall(ax)
        phi = float(self.info.get("injectionLocation", 0))
        self.bestFits[evalTime]["radDist"].plotCrossSection(phi=np.deg2rad(phi), ax=ax)
        ax.set_title(f"Injection location = {phi:.2f} degrees")

        for bolo_ in bolometerGroups:
            count_ += 1
            ax = f.add_subplot(2, num_columns, count_)
            self._plot_signal_panel(ax, bolo_, evalTime, units_label, legend=True)

        # --- Plot the radiation behavior
        tpf_ax = f.add_subplot(2, num_columns, count_ + 1)
        self._plot_tpf_panel(tpf_ax, evalTime)

        plt.tight_layout()

        if save:
            if "shot" in self.info and "tokamakName" in self.info:
                filename = f"{self.info['shot']}_{evalTime:.4f}.png"
                img_dir = (
                    EMIS3D_INPUTS_DIRECTORY
                    / self.info["tokamakName"]
                    / "runs"
                    / str(self.info["shot"])
                    / "images"
                )

                # --- Make the directory
                os.makedirs(img_dir, exist_ok=True)
                out_path = img_dir / filename
                plt.savefig(out_path, dpi=100, format="png")
                logger.debug(f"→ Figure saved to {out_path}")

        else:
            plt.show()

    def _plot_chord_panel(self, ax, tok, boloGroupName: str, evalTime: float) -> None:
        """Plots the first wall, one camera's chords and the radDist cross-section
        at that camera's toroidal location
        """

        tok._plot_first_wall(ax)
        tok._plot_bolometers(ax, boloGroupName)

        # Loop over each bolometer to find one that has the same group name
        # This is needed to get the phi location
        for bolo_ in tok.bolometers:
            # --- Add the radDist plot
            if bolo_.group_name == boloGroupName:

                phi = tok.get_ave_bolometer_tor_loc(boloGroupName=bolo_.group_name)
                if phi is not None:
                    self.bestFits[evalTime]["radDist"].plotCrossSection(
                        phi=np.deg2rad(phi), ax=ax
                    )
                break

        ax.set_title(boloGroupName)

    def _plot_signal_panel(
        self, ax, boloName: str, evalTime: float, units_label: str, legend: bool
    ) -> None:
        """Plots the observed signals, each fitted emission componenet, and their
        total for one bolometer

        Channels with the DEAD_CHANNEL_ERROR fit weight are excluded from the error bars
        and y-limit
        """
        data = np.asarray(self.fitData[evalTime]["boloData"][boloName], dtype=float)
        err = np.asarray(
            self.fitData[evalTime]["boloData_error"][boloName], dtype=float
        )
        channels = self._channel_numbers(boloName)

        # --- Mask the dead-channel fit weights for display
        valid = err != DEAD_CHANNEL_ERROR
        plot_err = np.where(valid, err, 0.0)

        ax.errorbar(
            channels,
            data,
            yerr=plot_err,
            marker="s",
            ms=5,
            c="black",
            linestyle="none",
            label="data",
        )

        # --- Fitted emission components and their total
        tot_emission = np.zeros(data.size)
        for jj, emissionName in enumerate(self.bestFits[evalTime]["synthData"]):
            em_data = np.asarray(
                self.bestFits[evalTime]["synthData"][emissionName][boloName],
                dtype=float,
            )
            tot_emission += em_data
            ax.plot(
                channels,
                em_data,
                marker=PLOT_MARKERS[jj % len(PLOT_MARKERS)],
                color=PLOT_COLORS[jj % len(PLOT_COLORS)],
                label=f"{emissionName} emission",
            )

        ax.plot(
            channels,
            tot_emission,
            color="purple",
            label="total emission",
            linewidth=3.0,
        )

        # --- y-limit from valid channels and the synthetic total only
        dat_ = np.concatenate((data + np.abs(err), tot_emission))
        ymax = np.nanmax(dat_)
        if np.any(valid):
            dat_ = np.concatenate(
                (data[valid] + np.abs(err[valid]), tot_emission[valid])
            )
            ymax = np.nanmax(dat_)

        if np.isfinite(ymax) and ymax > 0:
            ax.set_ylim(0, float(ymax) * 1.02)

        ax.set_xlabel("Channel Number")
        ax.set_ylabel(f"Emission {units_label}")
        ax.set_title(boloName)
        if legend:
            ax.legend(fontsize=8)

    def _plot_tpf_panel(self, ax, evalTime: float) -> None:
        """Plots the toroidal radiation distribution and peaking factor"""

        rad_distribution = self.bestFits[evalTime]["radiation_distribution"]

        y_data = rad_distribution["total_power"]
        y_max = np.nanmax(y_data)
        scale = np.floor(np.log10(y_max))
        y_scaled = y_data / 10**scale

        ax.plot(
            np.rad2deg(rad_distribution["phi"]),
            y_scaled,
            color="black",
            linewidth=2.0,
        )
        ax.set_ylim(
            np.floor(np.nanmin(y_scaled)),
            np.ceil(np.nanmax(y_scaled)),
        )
        ax.axvline(
            np.rad2deg(rad_distribution["peak_emission"]),
            np.floor(np.nanmin(y_scaled)),
            np.ceil(np.nanmax(y_scaled)),
            linestyle="dashed",
            color="tab:red",
        )
        ax.text(
            np.rad2deg(rad_distribution["peak_emission"]),
            np.floor(np.nanmin(y_scaled)) * 1.1,
            "Peak\nradiation",
            ha="center",
            va="bottom",
            size=10,
        )

        ax.set_xlabel("phi [degrees]")
        ax.set_ylabel(f"radiation [$10^{{{int(scale)}}}$ arb]")
        ax.set_title(
            f"time = {evalTime:.4f} ms, "
            f"TPF: {rad_distribution['toroidal_peaking_factor']:.2f}"
        )

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
