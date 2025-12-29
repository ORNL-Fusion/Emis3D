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

Re-organized pretty much everything during the refactor -JLH Aug., 2025


REMINDERS:
1. Make sure that the pre-processed SXR/Bolometer data are in the same units as those when the radDists were created
2. The bolometers have different responsivities with respect to each other! Some pre-analysis of the bolometer data is necessary
in order to scale them relative to each other. Then you should load the processed data in this program.
3. This program uses a right-handed coordinate system (positive phi in counter-clockwise direction when looking down).
So you need to offset your angles by 360 - x from DIII-D coordinates.



TODO:
1. Prepare fits -> Write definition to combine multiple locationDependent values together,
see _combine_synthetics_for_fits as a older starting point
2. Give the user the option to use the new error technique or use the error from the data
3. Double check field line tracer with output from MOFAT

Biggest Issues:
1. Find good radDist functions that represent what is going on
    - Have the helical distribution change orientation (and shape?) as it goes around the vessel
    - Add a tomography radDist mapping function (like BOLT?)
2. Implement a toroidal distribution function that is not symmetric around the injection loction
3. Re-vist how error is calculated for the observed data

"""

import os

import matplotlib.pyplot as plt
import numpy as np
from lmfit import minimize, report_fit

# from uncertainties import unumpy as unp
import main.Util_emis3D as Util_emis3D
from main.Globals import *
from main.Tokamak import Tokamak
from main.Util import (
    config_loader,
    convert_arrays_to_list,
    find_max_nested_lists,
    get_filenames_in_directory,
    read_h5,
)
from main.radDistFitting import RadDistFitting
from main.radDist import Helical, ElongatedRing
from scipy.integrate import simpson
import time
import flammkuchen as fl


class Emis3D:

    def __init__(
        self,
        tokamakName=None,
        runConfigName=None,
        initialize=True,
        verbose=False,
    ):
        """
        Main class for running the emis3D program

        tokamakName: str, the name of the tokamak
        runConfigName: str, the name of the run configuration file
        runner: bool, if True, it will run the program
        debug: bool, if True, it will print debug information
        """

        # --- Initialize variables
        self.data = {}
        self.info = None
        self.verbose = verbose
        self.error_free = True

        if initialize:
            self._initialize(tokamakName=tokamakName, runConfigName=runConfigName)

    def _initialize(self, tokamakName=None, runConfigName=None) -> None:
        """
        Runs the program
        """
        if not self.error_free:
            print("An error occurred, cannot run the program")
            return

        if tokamakName is None or runConfigName is None:
            print("tokamakName or runConfigName is None, cannot run the program")
            return

        self._load_config_file(tokamakName=tokamakName, runConfigName=runConfigName)
        self._load_bolometer_data()
        self._create_master_channel_order()
        self._load_radDists()

    def _perform_fits(self, evalTime=0.0, crossCalib=False) -> None:
        """
        Prepares the data, preforms the fits

        """
        if self.error_free:
            self._prepare_fits(evalTime=evalTime, crossCalib=crossCalib)
            t_start = time.time()

            """
            if crossCalib:
                self._minimize_radDists(evalTime=evalTime)
            else:
                if (
                    self.info is not None
                    and self.info.get("numProcessorsFitting", 1) > 1
                ):
                    self._run_parallel(
                        evalTime=evalTime, max_workers=self.info["numProcessorsFitting"]
                    )
                else:
            """

            # self._minimize_radDists(evalTime=evalTime)
            print(f"Fitting done in {time.time() - t_start:.2f} seconds")
            self._post_process_fit_arrangement(evalTime=evalTime)
            self._post_process_radiation_distribution(evalTime=evalTime)
            self._post_process_calculations(evalTime=evalTime)

    def _load_config_file(
        self, tokamakName=None, runConfigName=None, pathFileName=None
    ) -> None:
        """
        Loads the configuration file for the given tokamak

        tokamakName: str, the name of the tokamak
        runConfigName: str, the name of the run configuration file
        pathFileName: str, the full path to the configuration file, if None it will be constructed from tokamakName and runConfigName
        """

        # --- Only run if error free
        if self.error_free:
            try:

                if (
                    pathFileName is None
                    and tokamakName is not None
                    and runConfigName is not None
                ):
                    pathFileName = join(
                        EMIS3D_INPUTS_DIRECTORY, tokamakName, "runs", runConfigName
                    )

                if pathFileName is None:
                    raise Exception(
                        "tokamakName, runConfigName, or pathFileName is None, cannot load config file"
                    )

                # --- Load the configuration file, if it exists
                if os.path.isfile(pathFileName):
                    self.info = config_loader(pathFileName, verbose=self.verbose)

                    # --- Raise exception if the file failed to load
                    if self.info is None:
                        raise Exception(
                            f"Could not load the configuration file: {pathFileName}"
                        )

                    # --- Store the tokamak and runConfig name
                    self.info["tokamakName"] = tokamakName
                    self.info["runConfigName"] = runConfigName
                else:
                    raise Exception(f"File does not exist: {pathFileName}")

            except Exception as e:
                print(f"An error occurred loading the config file:\n{e}")
                self.error_free = False

    def _load_radDists(self) -> None:
        """
        Loads each radDist within the directories listed in the config file:
        radDistDirectories_LocIndependent, and radDistDirectories_LocDependent
        """

        # --- Only continue if error free
        if self.error_free:
            if self.verbose:
                print("Loading radDists")

            try:
                # --- Initilize synthetic signal arrays
                self.data.update(
                    {"synthetic": {"locDependent": {}, "locIndependent": {}}}
                )

                # --- Find the files, store as a nested list
                self.files = []

                if (
                    self.info is None
                    or "radDistDirectories_LocIndependent" not in self.info
                    or "radDistDirectories_LocDependent" not in self.info
                ):
                    print("No radDistDirectories found in the config file")
                    return

                count_ = -1
                # --- Load radDists that are location independent
                if "radDistDirectories_LocIndependent" in self.info:
                    dirs_ = self.info["radDistDirectories_LocIndependent"]
                    if len(dirs_) > 0:
                        for dir_ in dirs_:
                            pathFileName = os.path.join(
                                EMIS3D_INPUTS_DIRECTORY,
                                self.info["tokamakName"],
                                "radDists",
                                dir_,
                            )
                            files_ = get_filenames_in_directory(pathFileName)

                            # --- Loop over the files and load the radDist
                            for file_ in files_:
                                try:
                                    pass
                                    count_ += 1
                                    temp_ = RadDistFitting(radDistPath=file_)
                                    self.data["synthetic"]["locIndependent"][
                                        count_
                                    ] = temp_
                                except Exception as e:
                                    print(f"Error loading radDist {file_}")
                                    print(f"error: {e}")

                count_ = 0
                # --- Load radDists that are location independent
                if "radDistDirectories_LocDependent" in self.info:
                    dirs_ = self.info["radDistDirectories_LocDependent"]
                    if len(dirs_) > 0:
                        for ii, dir_ in enumerate(dirs_):
                            self.data["synthetic"]["locDependent"][f"loc_{ii}"] = {}

                            pathFileName = os.path.join(
                                EMIS3D_INPUTS_DIRECTORY,
                                self.info["tokamakName"],
                                "radDists",
                                dir_,
                            )
                            files_ = get_filenames_in_directory(pathFileName)

                            # --- Loop over the files and load the radDist
                            for file_ in files_:
                                try:
                                    pass
                                    count_ += 1
                                    temp_ = RadDistFitting(radDistPath=file_)
                                    self.data["synthetic"]["locDependent"][f"loc_{ii}"][
                                        count_
                                    ] = temp_
                                except Exception as e:
                                    print(f"Error loading radDist {file_}")
                                    print(f"error: {e}")
                if self.verbose:
                    print("Done loading radDists")

            except Exception as e:
                print(f"An error occured while loading synthetic data, {e}")
                self.error_free = False

    def _load_bolometer_data(self) -> None:
        """
        Loads the pre-calibrated SXR/bolometer data found in
        inputs/{tokamakName}/sxrData/

        It will use the filename found in the runConfig file (dataFileName)
        """

        # --- Only run if error free
        if self.error_free:
            try:
                # --- Exit if there is no config file loaded
                if self.info is None or "BOLOMETERS" not in self.info:
                    raise Exception("No BOLOMETERS found in the config file")

                # --- Load the data
                self.data["observed"] = {}
                for bolo_ in self.info["BOLOMETERS"]:
                    pathFileName = os.path.join(
                        EMIS3D_INPUTS_DIRECTORY,
                        self.info["tokamakName"],
                        "sxrData",
                        self.info["BOLOMETERS"][bolo_]["dataFileName"],
                    )
                    if os.path.isfile(pathFileName):
                        if self.verbose:
                            print(f"Loading bolometer data: {pathFileName}")
                        temp_ = read_h5(pathFileName)

                        # --- Convert channelOrder to a string array
                        for ii, ch_ in enumerate(temp_["channelOrder"]):
                            temp_["channelOrder"][ii] = ch_.decode("utf-8")

                        # --- Apply scaling factor if it exists
                        temp_["DATA_CALIBRATED"] *= self.info["BOLOMETERS"][bolo_].get(
                            "scalingFactor", 1.0
                        )

                        # --- Store the data
                        self.data["observed"][bolo_] = temp_
                    else:
                        raise FileNotFoundError(f"File does not exist: {pathFileName}")

            except Exception as e:
                print(f"An error occurred loading the bolometer data:\n{e}")
                self.error_free = False

    def _create_master_channel_order(self) -> None:
        """
        Creates a master channel list which all radDists and raw data
        will be organized from. This is based off the channelOrder within
        the loaded bolometer data
        """

        # --- Only run if error free
        if self.error_free:
            try:
                if self.info is None or "BOLOMETERS" not in self.info:
                    raise Exception("No BOLOMETERS found in the config file")

                # --- Initilize the arrays
                self.channel_order = {}
                self.channel_order["bolometer_order"] = []
                self.channel_order["channel_list"] = []

                for bolo in self.data["observed"]:
                    self.channel_order["bolometer_order"].append(bolo)
                    temp_ = []
                    for channel in self.data["observed"][bolo]["channelOrder"]:
                        temp_.append(channel)
                    self.channel_order["channel_list"].append(temp_)

            except Exception as e:
                print(f"An error occurred creating the master channel order:\n{e}")
                self.error_free = False

    def _syntheticSignalPreProcess(self) -> None:
        """
        It will then create create a nested list of the Radiance/Power,
        Radiance_error/Power_error, and distance arrays (if availble).

        If no data is availble, a zero will be placed for the radiance and an high error of 1.0e6
        will be put in the array

        This will be created within:
        self.data['synthetic']['injection_location_X']['fitData']
        self.data['synthetic']['injection_location_X']['fitData_error']
        self.data['synthetic']['injection_location_X']['scaleFactor']

        This is quite the mess since we have so many nested dictionaries and
        options to have multiple injection locations as well as multiple directions
        for each radDist (e.g. helical)

        Example, it stores stuff like this:
        self.data['synthetic']['injection_location_0'][*radDist number*]['fitData']['clockwise']['data'][*nested list of each bolometerGroup*]
        self.data['synthetic']['injection_location_0'][*radDist number*]['fitData']['clockwise']['data_error'][*nested list of each bolometerGroup*]
        """

        # --- Loop over each injection location
        for injection_loc in self.data["synthetic"]:

            # --- Loop over each radDist
            for num_ in self.data["synthetic"][injection_loc]:
                print(self.files[0][num_])
                units = self.data["synthetic"][injection_loc][num_]["info"]["units"]
                data = self.data["synthetic"][injection_loc][num_]["data"][units]
                data_err = self.data["synthetic"][injection_loc][num_]["data"][
                    f"{units}_error"
                ]
                scale_factor = self.data["synthetic"][injection_loc][num_]["data"][
                    "scaleFactor"
                ]

                # --- Loop over each type of radDist (e.g. clockwise, counterClock)
                self.data["synthetic"][injection_loc][num_]["fitData"] = {}
                fit_data = self.data["synthetic"][injection_loc][num_]["fitData"]

                for dir_ in data.keys():
                    if dir_.lower() != "channelorder":
                        # --- Create the blank lists
                        if dir_ not in fit_data:
                            fit_data[dir_] = {}
                            fit_data[dir_]["data"] = []
                            fit_data[dir_]["data_error"] = []
                            fit_data[dir_]["scaleFactor"] = []

                        # --- Loop over each bolometer group within the master list
                        for ii, boloGroup in enumerate(
                            self.channel_order["bolometer_order"]
                        ):
                            data_map = {}
                            data_map_error = {}
                            data_map_scale = {}

                            for bolo_ in boloGroup:
                                # --- Map out each data value within the boloGroup
                                if bolo_ in data[dir_]:
                                    # --- Synthetic data ---
                                    data_raw = data[dir_][bolo_]
                                    map_ = dict(
                                        zip(data["channelOrder"][bolo_], data_raw)
                                    )
                                    data_map.update(map_)

                                    # --- Synthetic data error ---
                                    data_raw_error = data_err[dir_][bolo_]
                                    map_error = dict(
                                        zip(data["channelOrder"][bolo_], data_raw_error)
                                    )
                                    data_map_error.update(map_error)

                                    # --- Scale factor ---
                                    data_scale_factor = scale_factor[dir_][bolo_]
                                    map_scale = dict(
                                        zip(
                                            data["channelOrder"][bolo_],
                                            data_scale_factor,
                                        )
                                    )
                                    data_map_scale.update(map_scale)
                            # ________________________________________
                            # ________________________________________
                            # ________________________________________
                            # --- Make a list of the data based on the master channel list
                            ordered_data = np.array(
                                [
                                    data_map.get(item, 0)
                                    for item in self.channel_order["channel_list"][ii]
                                ],
                                dtype=float,
                            )
                            # --- Convert to a list and store the result
                            ordered_data_list = convert_arrays_to_list(ordered_data)
                            fit_data[dir_]["data"].append(ordered_data_list)
                            # ________________________________________
                            # ________________________________________
                            # ________________________________________
                            # --- Add data error to fit_dadta
                            ordered_data_error = np.array(
                                [
                                    data_map_error.get(item, 1.0e3)
                                    for item in self.channel_order["channel_list"][ii]
                                ],
                                dtype=float,
                            )
                            # --- Convert to a list
                            ordered_data_error_list = convert_arrays_to_list(
                                ordered_data_error
                            )
                            fit_data[dir_]["data_error"].append(ordered_data_error_list)
                            # ________________________________________
                            # ________________________________________
                            # ________________________________________
                            # --- Add scale factor to fit_dadta
                            ordered_data_scale = np.array(
                                [
                                    data_map_scale.get(item, 1.0)
                                    for item in self.channel_order["channel_list"][ii]
                                ],
                                dtype=float,
                            )
                            # --- Convert to a list
                            ordered_data_scale_list = convert_arrays_to_list(
                                ordered_data_scale
                            )
                            fit_data[dir_]["scaleFactor"].append(
                                ordered_data_scale_list
                            )
                            # ________________________________________
                            # ________________________________________
                            # ________________________________________

    def _average_observed_data(self, arrayName="", evalTime=None) -> list:
        """
        Averages SXR/bolometer data over at the evalTime over the input dt

        Inputs:

        arrayName: str, The name of the bolometer array
        evalTime: float, the time to average the data
        """

        if evalTime is not None:
            time_ = self.data["observed"][arrayName]["TIME"]
            if self.info is not None and "dt" in self.info:
                dt_ = self.info["dt"]

                start = np.abs(time_ - (evalTime - dt_)).argmin()
                end = np.abs(time_ - (evalTime + dt_)).argmin()

                vals = np.mean(
                    self.data["observed"][arrayName]["DATA_CALIBRATED"][:, start:end],
                    axis=1,
                )

                # --- Check for NaNs
                temp_ = []
                for val in vals:
                    if np.isnan(val):
                        temp_.append(0)
                    else:
                        temp_.append(val)
                """
                vals_std  = np.std(
                    self.data["observed"][arrayName]["DATA_CALIBRATED"][:, start:end],
                    axis=1,
                )
                """
                ans = convert_arrays_to_list(temp_)
            else:
                ans = [1] * self.data["observed"][arrayName]["NUM_CHANNELS"]

        else:
            ans = [0] * self.data["observed"][arrayName]["NUM_CHANNELS"]

        return ans

    def _prepare_data_for_fit(self, evalTime=0.0) -> None:
        """
        Prepares the data for fitting by averaging data over the dt window (in the config file)
        and organzing the data in nested lists based on the master channel order

        evalTime: float, time of the fit

        Creates:
        self.fitData[evalTime][observed]        :: Averaged calibrated data
        self.fitData[evalTime][observed_error]  :: Error in the averaged calibrated data
        """

        # --- Only run if error free
        if self.error_free:
            try:
                if not hasattr(self, "fitData"):
                    self.fitData = {}

                self.fitData[evalTime] = {}
                self.fitData[evalTime]["observed"] = []
                self.fitData[evalTime]["observed_error"] = []

                # --- Average and map the data to a dict
                data_ = {}
                for bolo_ in self.data["observed"]:
                    temp = self._average_observed_data(
                        arrayName=bolo_, evalTime=evalTime
                    )

                    map_ = dict(zip(self.data["observed"][bolo_]["channelOrder"], temp))
                    data_.update(map_)

                self.fitData[evalTime]["dataMap"] = data_

                # --- Arrange the data in the same order as the synthetic data is ordered
                for channels in self.channel_order["channel_list"]:
                    temp = []
                    temp_e = []
                    # --- Find the max value for that array
                    max_ = 0
                    for channel in channels:
                        if data_[channel] > max_:
                            max_ = data_[channel]
                    for channel in channels:
                        temp.append(data_[channel])
                        if data_[channel] == 0:
                            temp_e.append(1.0e4)
                        else:
                            err_ = 0.01 * max_ / np.sqrt((data_[channel]) / max_)
                            temp_e.append(err_)

                    self.fitData[evalTime]["observed"].append(temp)
                    self.fitData[evalTime]["observed_error"].append(temp_e)

                # --- Also store the data for each bolometer, to make plotting easier
                self.fitData[evalTime]["boloData"] = {}
                self.fitData[evalTime]["boloData_error"] = {}
                max_ = 1.0e-6
                for bolo_ in self.data["observed"]:
                    self.fitData[evalTime]["boloData"][bolo_] = []
                    self.fitData[evalTime]["boloData_error"][bolo_] = []
                    for ch in self.data["observed"][bolo_]["channelOrder"]:
                        if data_[ch] > max_:
                            max_ = data_[ch]
                    for ch in self.data["observed"][bolo_]["channelOrder"]:
                        self.fitData[evalTime]["boloData"][bolo_].append(data_[ch])

                # --- Error cacluation, based on the maximum value observed within all of the data
                # and follows Poisson statistics
                for bolo_ in self.data["observed"]:
                    for ch in self.data["observed"][bolo_]["channelOrder"]:
                        # --- Avoid dividing by zero
                        if data_[ch] > 1.0:
                            err_ = 0.03 * max_ / np.sqrt((data_[ch]) / max_)
                        else:
                            err_ = 1.0e4
                        self.fitData[evalTime]["boloData_error"][bolo_].append(err_)

                if self.verbose:
                    print(f"Observed data prepared for fitting")

            except Exception as e:
                print(f"An error occured while preparing data for the fit: {e}")
                self.error_free = False

    def _prepare_synthetic_for_fits(self, evalTime=0.0, crossCalib=False) -> None:
        """
        Arranges the synthetic data
        """
        # --- Only run if error free
        if self.error_free:
            try:
                print("Preparing synthetic data for fitting")
                max_data_val = find_max_nested_lists(self.fitData[evalTime]["observed"])
                if self.info is not None and "varyScaleFactor" in self.info:

                    # TODO: Shorten this since there is repeat code!!
                    # --- Arrange and create parameters for the location dependent data
                    for loc in self.data["synthetic"]["locDependent"]:
                        for number_ in self.data["synthetic"]["locDependent"][loc]:
                            radD = self.data["synthetic"]["locDependent"][loc][number_]
                            radD.prepare_for_fits(
                                self.channel_order["channel_list"],
                                data_max=max_data_val,
                            )
                            boloNames = None
                            if crossCalib:
                                boloNames = self.channel_order["bolometer_order"]
                            radD.create_parameters(
                                boloNames=boloNames,
                                varyScaleFactor=self.info["varyScaleFactor"],
                            )

                    # --- Arrange and create parameters for the location independent data
                    for number_ in self.data["synthetic"]["locIndependent"]:
                        radD = self.data["synthetic"]["locIndependent"][number_]
                        radD.prepare_for_fits(
                            self.channel_order["channel_list"], data_max=max_data_val
                        )
                        boloNames = None
                        if crossCalib:
                            boloNames = self.channel_order["bolometer_order"]
                        radD.create_parameters(
                            boloNames=boloNames,
                            varyScaleFactor=self.info["varyScaleFactor"],
                        )

                if self.verbose:
                    print("Done preparing synthetic data for fit")

            except Exception as e:
                print(
                    f"An error occured while preparing synthetic data for fitting: {e}"
                )
                self.error_free = False

    def _prepare_fits(self, evalTime=0.0, crossCalib=False) -> None:
        """
        Prepares the data and synthetic signals for fitting

        evalTime: float, time to preform the fit
        """
        # --- Only run if error free
        if self.error_free:
            try:
                if self.verbose:
                    print("Preparing data for fitting")

                self._prepare_data_for_fit(evalTime=evalTime)
                self._prepare_synthetic_for_fits(
                    evalTime=evalTime, crossCalib=crossCalib
                )

                if self.verbose:
                    print("Arranging radDists for fitting")

                if not hasattr(self, "fits"):
                    self.fits = {}

                # --- Initilze the fitting dictionary
                self.fits[evalTime] = {}
                fitCount = -1

                # --- Loop over each location independent radDist
                if len(self.data["synthetic"]["locIndependent"]) > 0:
                    for number in self.data["synthetic"]["locIndependent"]:
                        fitCount += 1
                        self.fits[evalTime][fitCount] = {}

                        radDist_ = self.data["synthetic"]["locIndependent"][number]
                        self.fits[evalTime][fitCount]["info"] = {}
                        self.fits[evalTime][fitCount]["info"]["radDists"] = {
                            "locationdependence": "locIndependent",
                            "radDistNumber": number,
                            "location": None,
                        }

                        self.fits[evalTime][fitCount]["synthetic_dict"] = {}
                        synthetic_dict = self.fits[evalTime][fitCount]["synthetic_dict"]
                        self.fits[evalTime][fitCount]["parameters"] = (
                            radDist_.fitSynthetic["params"]["params"]
                        )

                        # --- Add paramName to the list
                        synthetic_dict["paramName"] = radDist_.fitSynthetic["params"][
                            "paramName"
                        ]
                        synthetic_dict["injectionLocation"] = self.data["synthetic"][
                            "locIndependent"
                        ][number].info["injectionLocation"]
                        synthetic_dict["injectionLocation_rad"] = np.deg2rad(
                            synthetic_dict["injectionLocation"]
                        )
                        synthetic_dict["emissionNames"] = radDist_.info["emissionNames"]

                        # --- Point the data in the radDist to the fit dict
                        for emissionName in radDist_.info["emissionNames"]:
                            synthetic_dict[emissionName] = {}
                            synthetic_dict[emissionName]["scaleSynth"] = (
                                radDist_.fitSynthetic[emissionName]["scaleSynth"]
                            )
                            synthetic_dict[emissionName]["scaleFactor"] = (
                                radDist_.fitSynthetic[emissionName]["scaleFactor"]
                            )
                            synthetic_dict[emissionName]["data"] = (
                                radDist_.fitSynthetic[emissionName]["data"]
                            )
                            synthetic_dict[emissionName]["data_error"] = (
                                radDist_.fitSynthetic[emissionName]["data_error"]
                            )

                # --- Loop over each location location dependent radDist
                if len(self.data["synthetic"]["locDependent"]) > 0:
                    for loc_ in self.data["synthetic"]["locDependent"]:
                        for number in self.data["synthetic"]["locDependent"][loc_]:
                            fitCount += 1
                            self.fits[evalTime][fitCount] = {}

                            radDist_ = self.data["synthetic"]["locDependent"][loc_][
                                number
                            ]
                            self.fits[evalTime][fitCount]["info"] = {}
                            self.fits[evalTime][fitCount]["info"]["radDists"] = {
                                "locationdependence": "locDependent",
                                "radDistNumber": number,
                                "location": loc_,
                            }
                            self.fits[evalTime][fitCount]["synthetic_dict"] = {}
                            synthetic_dict = self.fits[evalTime][fitCount][
                                "synthetic_dict"
                            ]

                            # --- Add each parameter to the fit array
                            self.fits[evalTime][fitCount]["parameters"] = (
                                radDist_.fitSynthetic["params"]["params"]
                            )
                            synthetic_dict["paramName"] = radDist_.fitSynthetic[
                                "params"
                            ]["paramName"]

                            synthetic_dict["injectionLocation"] = self.data[
                                "synthetic"
                            ]["locDependent"][loc_][number].info["injectionLocation"]
                            synthetic_dict["injectionLocation_rad"] = np.deg2rad(
                                synthetic_dict["injectionLocation"]
                            )
                            synthetic_dict["emissionNames"] = radDist_.info[
                                "emissionNames"
                            ]
                            # --- Point the data in the radDist to the fit dict
                            for emissionName in radDist_.info["emissionNames"]:
                                synthetic_dict[emissionName] = {}
                                synthetic_dict[emissionName]["scaleSynth"] = (
                                    radDist_.fitSynthetic[emissionName]["scaleSynth"]
                                )
                                synthetic_dict[emissionName]["scaleFactor"] = (
                                    radDist_.fitSynthetic[emissionName]["scaleFactor"]
                                )
                                synthetic_dict[emissionName]["data"] = (
                                    radDist_.fitSynthetic[emissionName]["data"]
                                )
                                synthetic_dict[emissionName]["data_error"] = (
                                    radDist_.fitSynthetic[emissionName]["data_error"]
                                )

                num_fits = len(self.fits[evalTime])
                if self.info is not None:
                    self.info["numFits"] = num_fits
                self.fits[evalTime]["chiSqVec"] = np.full(num_fits, 1.0e6)

            except Exception as e:
                print(f"An error occured while preparing the fits: {e}")
                self.error_free = False

    def _combine_synthetics_for_fits(self, evalTime=0.0):
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

    def _minimize_radDists(self, evalTime=0.0, crossCalib=False):
        """
        Performs fits, current uses a for loop, can be modified to run in parallel
        """

        # --- Only run if error free
        if self.error_free:
            try:
                if self.info is not None and "scale_def" in self.info:
                    # --- Data used for fitting
                    data_dict = self.fitData[evalTime]

                    for ii in self.fits[evalTime]:
                        if type(ii) == int:
                            if ii % 1_000 == 0 and self.verbose:
                                print(
                                    f"Preforming fit {ii} out of {self.info['numFits']}"
                                )
                            synth_dict = self.fits[evalTime][ii]["synthetic_dict"]
                            pars = self.fits[evalTime][ii]["parameters"]

                            try:
                                boloNames = None
                                # --- Include bolometer names if doing a cross-calibration
                                if crossCalib:
                                    if (
                                        self.channel_order is not None
                                        and "bolometer_order" in self.channel_order
                                    ):
                                        boloNames = self.channel_order[
                                            "bolometer_order"
                                        ]
                                    else:
                                        boloNames = None

                                residual = True
                                self.fits[evalTime][ii]["fit"] = minimize(
                                    Util_emis3D.residual,
                                    pars,
                                    args=(
                                        data_dict,
                                        synth_dict,
                                        self.info["scale_def"],
                                        boloNames,
                                        residual,
                                    ),
                                    method="leastsq",
                                )

                                self.fits[evalTime]["chiSqVec"][ii] = self.fits[
                                    evalTime
                                ][ii]["fit"].chisqr.item()
                            except Exception as e:
                                print(
                                    f"An error occured during the {ii} iteration: {e}"
                                )
                                self.fits[evalTime]["chiSqVec"][ii] = 1.0e6
            except Exception as e:
                print(f"An error occured while fitting: {e}")

    def _rebuild_radDist(
        self,
        locationDependence="locIndependent",
        radDistNumber=0,
        location=None,
        bestFit=False,
        evalTime=0.0,
    ):
        """
        Will rebuild the radDist from the radDist class in order to plot them.
        It will ignore actually calculating the powerPerBin and observation of
        each bolometer array

        Parameters:

        locationDependence : str
                             either locIndependent or locDependent
        radDistNumber      : int
                             the radDist number under self.data['synthetic']['locationDependent'][loc] or
                             self.data['synthetic']['locationIndependent'][loc]
        location           : str
                             The location, only needed to locationDependent values
        bestFit            : boolean
                             Set to true to just get the radDist from the best fit
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
        if info["distType"] == "helical":
            radDist_ = Helical(setFieldLine=False)
        elif info["distType"] == "elongatedRing":
            radDist_ = ElongatedRing()
        else:
            print(f"self_rebuild_radDist() only supports Helical or ElongatedRing")
            return

        # --- Update the information
        if radDist_.info is not None:
            for val in info:
                if radDist_.info is None:
                    radDist_.info = {}
                radDist_.info[val] = info[val]

        # --- Update the data
        radDist_.data = {}
        for key in rad_.data:
            radDist_.data[key] = rad_.data[key]

        # --- Build the tokamak
        radDist_._build_tokamak(
            tokamakName=radDist_.info["tokamakName"],
            mode="Build",
            reflections=False,
            eqFileName=radDist_.info["eqFileName"],
        )

        # --- Build the field line, if it is a helical distribution
        if type(radDist_) == Helical:
            radDist_.setFieldLine()

        return radDist_

    def _post_process_fit_arrangement(self, evalTime=0.0, crossCalib=False):
        """
        Arranges synthetic data back into the bolometer array format
        """

        # --- Find the best fit
        bestFitID = np.array(self.fits[evalTime]["chiSqVec"]).argmin().item()

        # --- Print the results of the best fit
        print("\n" * 2)
        report_fit(self.fits[evalTime][bestFitID]["fit"])
        print("\n" * 2)

        # --- Store the best fit
        if not hasattr(self, "bestFits"):
            self.bestFits = {}

        self.bestFits[evalTime] = self.fits[evalTime][bestFitID]
        self.bestFits[evalTime]["bestFitID"] = bestFitID

        if self.info is None or "scale_def" not in self.info:
            print("No scale_def found in the config file")
            return

        boloNames = None
        if crossCalib:
            if (
                self.channel_order is not None
                and "bolometer_order" in self.channel_order
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
                map_ = dict(
                    zip(self.channel_order["channel_list"][ii], data_[emissionName][ii])
                )
                temp_dict.update(map_)

            # --- Loop over the bolometer and channels to build the lists
            for bolo_ in self.data["observed"]:
                self.bestFits[evalTime]["synthData"][emissionName][bolo_] = []
                for ch in self.data["observed"][bolo_]["channelOrder"]:
                    self.bestFits[evalTime]["synthData"][emissionName][bolo_].append(
                        temp_dict[ch]
                    )

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
            bestFit=True, evalTime=evalTime
        )

    def _post_process_radiation_distribution(self, evalTime=0.0):
        """
        Calculates the radiation amplitude distribution from the best fit
        """

        if self.info is not None:
            scale_def = self.info["scale_def"]
        else:
            return

        radDist_ = self.bestFits[evalTime]["radDist"]
        self.bestFits[evalTime]["radiation_distribution"] = {}
        rad_distribution = self.bestFits[evalTime]["radiation_distribution"]

        params = self.bestFits[evalTime]["fit"].params.valuesdict()

        mu = self.bestFits[evalTime]["synthetic_dict"]["injectionLocation_rad"]
        mu_deg = self.bestFits[evalTime]["synthetic_dict"]["injectionLocation"]
        scale_def = self.info["scale_def"]
        emissionNames = self.bestFits[evalTime]["synthetic_dict"]["emissionNames"]
        numTransits = len(self.bestFits[evalTime]["radDistInfo"]["emissionNames"]) / 2.0

        # --- Lists to fit the whole radiation distribution too
        x_all = []
        y_all = []

        if radDist_.info is not None and "distType" in radDist_.info:
            if radDist_.info["distType"] == "helical":
                emissionNames = ["clockwise", "counterClock"]
            else:
                emissionNames = radDist_.info["emissionNames"]

        for emissionName in emissionNames:

            # --- Fit assumed phi - mu = 0, aka center is on the injection location
            dphi = np.linspace(-np.pi, np.pi, 200)

            rad_distribution[emissionName] = {}

            a = params[f"a_{mu_deg}"]
            b = params[f"b_{emissionName}_{mu_deg}"]

            scale_ = Util_emis3D.scale_wrapper(
                a=a,
                b=b,
                phi=np.zeros(1),
                dphi=dphi,
                mu=0.0,
                scale_def=scale_def,
                emissionName=emissionName,
            )

            if "clockwise" in emissionName:
                loc_ = dphi <= 0
                # Scale back up to 2 pi since the helical distribution is a full revolutions
                dphi_scale = 2.0 * numTransits

            elif "counterClock" in emissionName:
                loc_ = dphi > 0
                dphi_scale = 2.0 * numTransits
            else:
                loc_ = np.full(dphi.shape[0], fill_value=True)
                dphi_scale = 1.0

            dphi *= dphi_scale

            phi_unwrapped = dphi[loc_] + mu
            amplitude_ = scale_[loc_]

            # --- Add them to the total distribution
            x_all.extend(phi_unwrapped)
            y_all.extend(amplitude_)

            # --- Wrap phi so it is from 0 to 2pi
            phi_wrapped = np.mod(phi_unwrapped, 2.0 * np.pi)
            sort_indx = np.argsort(phi_wrapped)

            # --- Populate the arrays
            rad_distribution[emissionName]["phi"] = phi_wrapped[sort_indx]
            rad_distribution[emissionName]["amplitude"] = amplitude_[sort_indx]
            rad_distribution[emissionName]["phi_unwrapped"] = phi_unwrapped

            rad_distribution[emissionName]["phi_left_handed_deg"] = np.rad2deg(
                2.0 * np.pi - phi_wrapped[sort_indx]
            )
            rad_distribution[emissionName]["amplitude_unwrapped"] = amplitude_

        # --- Preform the fits
        rad_distribution["total"] = {}
        rad_distribution["total"]["phi"] = x_all
        rad_distribution["total"]["amp"] = y_all

    def _post_process_calculations(self, evalTime=0.0) -> None:
        """
        Rebuilds the radDist, calculates the total powerPerBin, calculates the
        toroidal peaking factor
        """

        # --- Only run if self._post_process_fit_arrangement() has been run
        if not hasattr(self, "bestFits"):
            print(
                "Please run self._post_process_fit_arrangement() prior to running self._post_process_calculations()"
            )
            return

        # --- Shorten some variable calls
        radDist_ = self.bestFits[evalTime]["radDist"]
        rad_distribution = self.bestFits[evalTime]["radiation_distribution"]
        emissionNames = radDist_.info["emissionNames"]
        self.bestFits[evalTime]["powerPerBin"] = {}
        powerPerBin = self.bestFits[evalTime]["powerPerBin"]

        # --- Find the phi location for each bin range
        numBins = radDist_.info["numBins"]
        phibin = np.linspace(0, 2.0 * np.pi, numBins + 1)
        phibin_center = (phibin[:-1] + phibin[1:]) / 2.0
        mu = self.bestFits[evalTime]["synthetic_dict"]["injectionLocation_rad"]

        x_all = []
        y_all = []

        # --- Unwrap and combine the powerPerBin for both radiation distribution functions
        for emissionName in emissionNames:
            powerPerBin[emissionName] = {}
            ppb_amp = np.array(radDist_.data["powerPerBin"][emissionName]).copy()

            # --- elongatedRing distributions are symmetric, so we can skip a lot of this
            if emissionName == "elongatedRing":
                # --- The values are constant, so I am just extending the arrays,
                # otherwise there are issues with the interpolation near the endpoints
                y_ = np.mean(ppb_amp)
                x_all.extend(np.linspace(0, 2.0 * np.pi, 20))
                y_all.extend(np.full(20, fill_value=y_))

            else:
                # --- Unwrap the powerPerBin
                dphi = phibin_center - mu

                # --- Determine the offset
                offset = 0.0
                if "rev" in emissionName:
                    offset = int(emissionName[-1]) * 2.0 * np.pi

                # --- Remove the data point at dphi = 0, it is always wrong for some reason
                indx = np.abs(dphi).argmin()

                dphi = np.delete(dphi, indx)
                ppb_amp = np.delete(ppb_amp, indx)

                if "counterClock" in emissionName:
                    dphi[dphi <= 0] += 2.0 * np.pi
                elif "clockwise" in emissionName:
                    dphi[dphi > 0] -= 2.0 * np.pi

                # --- Arrange the data in ascending order
                sort_ = np.argsort(np.array(dphi))
                dphi_ = dphi[sort_] + mu

                # --- Add the offset
                dphi_ += offset

                # --- Clockwise data should be negative
                if "clockwise" in emissionName:
                    dphi_ *= -1.0

                # --- Add mu back, then put it in the master array
                x_all.extend(dphi_)
                y_all.extend(ppb_amp[sort_])

            # --- Arrange the data in ascending order
            sort_ = np.argsort(np.array(x_all))
            x_all_ppb = np.array(x_all)[sort_]
            y_all_ppb = np.array(y_all)[sort_]

            # --- Perform fits on the data
            x_min, x_max = np.min(x_all_ppb), np.max(x_all_ppb)
            phi_ = np.linspace(x_min, x_max, 500)

            # --- Fit the powerPerBin
            ppb_fit = np.interp(phi_, x_all_ppb, y_all_ppb)

            # --- Fit the radiation distribution
            x_pts = rad_distribution["total"]["phi"].copy()
            y_pts = rad_distribution["total"]["amp"].copy()
            # --- Wrap the distribution if it is only from -pi to pi
            if np.max(x_pts) - np.min(x_pts) == 2.0 * np.pi:
                x_pts = np.array(x_pts) % (2.0 * np.pi)
                sort_ = np.argsort(x_pts)
                x_pts = x_pts[sort_]
                y_pts = np.array(y_pts)[sort_]

            y_rad_distr = np.interp(phi_, x_pts, y_pts)

            # --- Find the synthetic scaling factor, it should be the same for each emissionName
            emissionName = emissionNames[0]
            scale_synth = self.bestFits[evalTime]["synthetic_dict"][emissionName][
                "scaleSynth"
            ]

            ppb_total = scale_synth * ppb_fit * y_rad_distr

            # --- Scale the results back to 0 to 360 degrees
            phi_wrapped = np.linspace(0, 2.0 * np.pi, 360)
            ppb_total_wrapped = np.zeros(phi_wrapped.shape[0])

            # --- Find all the equivilent data within the range
            for ii, theta in enumerate(phi_wrapped):
                ks = np.arange(
                    (x_min - theta) // (2.0 * np.pi),
                    (x_max - theta) // (2.0 * np.pi) + 1,
                )
                x_vals = theta + 2.0 * np.pi * ks

                # --- Keep only the values within the range
                x_vals = x_vals[(x_vals >= x_min) & (x_vals <= x_max)]

                ppb_total_wrapped[ii] = np.sum(np.interp(x_vals, phi_, ppb_total))

            # --- Store the results
            powerPerBin["total"] = {}
            powerPerBin["total"]["phi_unwrapped"] = phi_
            powerPerBin["total"]["powerPerBin_unwrapped"] = ppb_total

            # --- Something funky happens with the elongatedRing distribution when they
            # are wrapped, so just keep the unwrapped version
            if emissionName == "elongatedRing":
                phi_wrapped = phi_
                ppb_total_wrapped = ppb_total

            powerPerBin["total"]["phi"] = phi_wrapped
            powerPerBin["total"]["powerPerBin"] = ppb_total_wrapped

            # --- Find the toroidal peaking factor
            x = phi_wrapped
            y = ppb_total_wrapped
            w_int = simpson(y, x=x) / (2.0 * np.pi)
            powerPerBin["total"]["toroidal_peaking_factor"] = np.max(y) / w_int

    def _plot_bestFit(self, evalTime=0.0, save=False) -> None:
        """
        Plots the fit synthetic signal, data, and radDist for the given evalTime
        """
        boloName = "Unknown"

        if self.info is None:
            return

        # --- Rebuild the bestFit radDist
        if not hasattr(self, "bestFits"):
            print(
                "Please run self._post_process_fit_arrangement() and self._post_process_calculations() prior to running self._plot_bestFit()"
            )

        # --- Find the eqFileName, if it exists
        eqFileName = self.bestFits[evalTime]["radDistInfo"]["eqFileName"]

        tok = Tokamak(
            tokamakName="DIII-D",  # self.info["tokamakName"],
            mode="Build",
            reflections=False,
            eqFileName=eqFileName,
            loadBolometers=True,
        )

        bolometers = self.info["BOLOMETERS"]
        num_columns = len(bolometers) + 1
        f = plt.figure(figsize=(15, 8))

        # --- Plot the bolometer chords and radDist contour
        count_ = 0
        for ii, boloName in enumerate(list(bolometers.keys())):
            count_ += 1
            f_top = f.add_subplot(2, num_columns, count_)
            tok._plot_first_wall(f_top)
            for bolo_ in tok.bolometers:
                if bolo_.info["GROUP_NAME"] == boloName:
                    tok._plot_bolometers(f_top, bolo_.name)

                    # --- Add the radDist plot
                    phi = np.deg2rad(bolo_.info["CAMERA_POSITION_R_Z_PHI"][2])
                    if hasattr(self, "bestFits"):
                        self.bestFits[evalTime]["radDist"].plotCrossSection(
                            phi=phi, ax=f_top
                        )
            f_top.set_title(boloName)

        # --- Plot the contour at the injection location
        count_ += 1
        f_top = f.add_subplot(2, num_columns, count_)
        tok._plot_first_wall(f_top)
        phi = 0
        if hasattr(self, "bestFits"):
            if self.info is not None and "injectionLocation" in self.info:
                phi = self.info["injectionLocation"]
                self.bestFits[evalTime]["radDist"].plotCrossSection(phi=phi, ax=f_top)
        f_top.set_title(f"Injection location = {phi:.2f} degrees")

        # --- Plot the observed emissivities
        colors = ["green", "orangered", "blue", "cyan", "magenta"]
        markers = ["^", "o", "s", "D", "v"]

        for ii, bolo_ in enumerate(list(bolometers.keys())):
            count_ += 1
            f_bottom = f.add_subplot(2, num_columns, count_)
            numChan = len(self.fitData[evalTime]["boloData"][bolo_])
            channels = np.arange(1, numChan + 1, 1)

            # --- Observed data
            f_bottom.errorbar(
                channels,
                self.fitData[evalTime]["boloData"][bolo_],
                yerr=self.fitData[evalTime]["boloData_error"][bolo_],
                marker="s",
                ms=5,
                c="black",
                linestyle="none",
                label="data",
            )

            tot_emission = []
            for jj, emissionName in enumerate(self.bestFits[evalTime]["synthData"]):
                if len(tot_emission) == 0:
                    tot_emission = np.array(
                        self.bestFits[evalTime]["synthData"][emissionName][bolo_]
                    )
                else:
                    tot_emission += np.array(
                        self.bestFits[evalTime]["synthData"][emissionName][bolo_]
                    )

                f_bottom.plot(
                    channels,
                    self.bestFits[evalTime]["synthData"][emissionName][bolo_],
                    marker=markers[jj],
                    color=colors[jj],
                    label=f"{emissionName} emission",
                )
                f_bottom.set_xlabel("Channel Number")
                f_bottom.set_ylabel("Emission [arb. units]")
                f_bottom.set_title(f"{bolo_}")

            f_bottom.plot(
                channels,
                tot_emission,
                color="purple",
                label="total emission",
            )

            # --- Set the lower y-axis bound to zero
            f_bottom.set_ylim(0, f_bottom.get_ylim()[1])

            if ii == 0:
                f_bottom.legend(fontsize=8)

        # --- Plot the radiation behavior
        tpf_plot = f.add_subplot(2, num_columns, count_ + 1)
        tpf_plot.plot(
            np.rad2deg(self.bestFits[evalTime]["powerPerBin"]["total"]["phi"]),
            self.bestFits[evalTime]["powerPerBin"]["total"]["powerPerBin"],
            color="black",
            linewidth=2.0,
        )
        tpf_plot.set_xlabel("phi [degrees]")
        tpf_plot.set_ylabel("radiation [arb]")
        tpf_plot.set_title(
            f"time = {evalTime:.1f} ms, TPF: {self.bestFits[float(evalTime)]["powerPerBin"]["total"]["toroidal_peaking_factor"]:.2f}"
        )

        plt.tight_layout()

        if save:
            if (
                self.info is not None
                and "shot" in self.info
                and "tokamakName" in self.info
            ):
                filename = f"{self.info['shot']}_{evalTime:.2f}.png"
                pathFileName = join(
                    EMIS3D_INPUTS_DIRECTORY,
                    self.info["tokamakName"],
                    "runs",
                    str(self.info["shot"]),
                    "images",
                )

                # --- Make the directory
                os.makedirs(pathFileName, exist_ok=True)

                pathFileName_ = join(pathFileName, filename)
                plt.savefig(pathFileName_, dpi=100, format="png")
                print(f"Figure saved to {pathFileName_}")

        else:
            plt.show()

    def _cleanup_fits(self, evalTime=0.0) -> None:
        """
        Preforms fitting of the synthetic data to the observed data
        over the times in evalTimes
        """

        if self.verbose:
            print(f"Deleting bad fits for = {evalTime:.2f} ms\n")

        # --- Delete the bad fits to save memory
        if hasattr(self, "fits"):
            if evalTime in self.fits:
                for ii in list(self.fits[evalTime].keys()):
                    if type(ii) == int:
                        if ii != self.bestFits[evalTime]["bestFitID"]:
                            del self.fits[evalTime][ii]

    def _save_bestFits(self) -> None:
        """
        Saves the best fits
        """
        if self.info is not None and "shot" in self.info and "tokamakName" in self.info:
            keys = list(self.bestFits.keys())
            if len(keys) > 1:
                min = np.min(keys)
                max = np.max(keys)
                filename = f"{self.info['shot']}_bestFits_{min:.2f}_to_{max:.2f}.h5"
            else:
                filename = f"{self.info['shot']}_bestFits_{keys[0]:.2f}.h5"

            pathFileName = join(
                EMIS3D_INPUTS_DIRECTORY,
                self.info["tokamakName"],
                "runs",
                str(self.info["shot"]),
                filename,
            )
            print(pathFileName)

            dict_ = {"fit_data": self.fitData, "bestFits": self.bestFits}
            fl.save(pathFileName, dict_)

            print(f"Best fits and fitData saved to: {pathFileName}")

    def _load_bestFits(self, path="") -> None:
        """
        Loads the best fits to do analysis
        """
        self.bestFits = {}
        self.fitData = {}
        temp = fl.load(path)
        if isinstance(temp, dict):
            for key in list(temp.keys()):
                for evalTime in temp[key]:
                    if key == "fit_data":
                        self.fitData[evalTime] = temp[key][evalTime]
                    elif key == "bestFits":
                        self.bestFits[evalTime] = temp[key][evalTime]
