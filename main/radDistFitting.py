# radDist_Fitting.py
"""
Contains the class used by emis3D for a given radDist

NOTE: The minimum values for the fitting are hard coded in self.create_parameters
This was done so the von mises distribution decays to zero for the helical distribution at +/- pi
which corresponds to +/- 720 degrees, since phi is scaled down during fitting.

Written by JLH Aug. 2025
"""

import numpy as np
from lmfit import Parameters
from main.Util import load_json, convert_arrays_to_list, find_max_nested_lists


class RadDistFitting:
    """
    radDist class used when fitting each radDist. This class preforms actions
    such as loading the data, organizing it based of a specific channel order,
    generating the initial parameters to be used while fitting and more.
    """

    def __init__(self, radDistPath=None):
        self.info = {}
        self.info["radDistPath"] = radDistPath

        if radDistPath is not None:
            self._load_radDist()
            self._map_signals()

    def _load_radDist(self) -> None:
        """
        Loads the radDist given by radDistPath
        """
        try:
            temp = load_json(self.info["radDistPath"])
            self.data = temp["data"]
            self.info.update(temp["info"])

        except Exception as e:
            print(
                f"An error occured while trying to load the file: {self.info['radDistPath']}: {e}"
            )

    def _map_signals(self) -> None:
        """
        Creates a dictionary that maps the channel name to the synthetic signal
        """
        units = self.info["units"]
        bolos = list(self.data[units]["channelOrder"].keys())
        self.maps = {}
        self.data_maps = {}
        for emissionName in self.info["emissionNames"]:
            self.data_maps[emissionName] = {}
            for val in ["scaleFactor", "data", "data_error"]:
                self.data_maps[emissionName][val] = {}

        # --- Loop over each signal
        for bolo_ in bolos:
            channels = self.data[units]["channelOrder"][bolo_]

            for emissionName in self.info["emissionNames"]:
                scale_ = self.data["scaleFactor"][emissionName][bolo_]
                map_scale = dict(zip(channels, scale_))

                data_ = self.data[units][emissionName][bolo_]
                map_data = dict(zip(channels, data_))

                data_error = self.data[f"{units}_error"][emissionName][bolo_]
                map_data_error = dict(zip(channels, data_error))

                self.data_maps[emissionName]["data"].update(map_data)
                self.data_maps[emissionName]["data_error"].update(map_data_error)
                self.data_maps[emissionName]["scaleFactor"].update(map_scale)

    def prepare_for_fits(self, channelOrder, data_max=None) -> None:
        """
        Prepares the data for fitting, will arrange the data
        in nested lists as well as create parameters used with LMFIT.

        channelOrder should be the nested lists from emis3D.channel_order['channel_list']
        data_max: float, the maximum value within the data, we will scale the synthetic data to this

        This dict will be read directly with the residual definition within
        Util_emis3D.

        The data is stored as:
        self.fitSynthetic[emissionName]
        self.fitSynthetic['params']
        """

        self.fitSynthetic = {}

        for emissionName in self.info["emissionNames"]:
            # --- Create the blank lists
            self.fitSynthetic[emissionName] = {}
            for val in ["scaleFactor", "data", "data_error"]:
                self.fitSynthetic[emissionName][val] = []

            for ch_list in channelOrder:
                temp_data = []
                temp_data_error = []
                temp_scale = []
                for chan in ch_list:
                    temp_data.append(self.data_maps[emissionName]["data"].get(chan, 0))
                    temp_data_error.append(
                        self.data_maps[emissionName]["data_error"].get(chan, 1.0)
                    )
                    temp_scale.append(
                        self.data_maps[emissionName]["scaleFactor"].get(chan, 1.0)
                    )

                # --- Flatten the list
                flatten = np.array(temp_scale).flatten()
                temp_scale = flatten.tolist()

                self.fitSynthetic[emissionName]["data"].append(temp_data)
                self.fitSynthetic[emissionName]["data_error"].append(temp_data_error)
                self.fitSynthetic[emissionName]["scaleFactor"].append(temp_scale)

        # --- Find the synthetic scaling factor
        if data_max is not None:
            scale_ = []
            for emissionName in self.info["emissionNames"]:
                synth_max = find_max_nested_lists(
                    self.fitSynthetic[emissionName]["data"]
                )
                scale_.append(data_max / synth_max)

            # --- Use the max value
            scale = np.nanmax(scale_)
        else:
            scale = 1.0

        # --- Scale the synthetic data
        for emissionName in self.info["emissionNames"]:
            self.fitSynthetic[emissionName]["scaleSynth"] = scale
            # --- Scale the data
            temp_ = []
            for val in self.fitSynthetic[emissionName]["data"]:
                temp_.append(scale * np.array(val))
            self.fitSynthetic[emissionName]["data"] = convert_arrays_to_list(temp_)

            # --- Scale the error
            temp_ = []
            for val in self.fitSynthetic[emissionName]["data_error"]:
                temp_.append(scale * np.array(val))
            self.fitSynthetic[emissionName]["data_error"] = convert_arrays_to_list(
                temp_
            )

    def create_parameters(self, boloNames=None, varyScaleFactor=False) -> None:
        """
        Creates the LMFIT parameters for the radDist
        """
        self.fitSynthetic["params"] = {}
        self.fitSynthetic["params"]["paramName"] = []
        params = Parameters()

        # --- Create parameters for the normal fitting case
        if boloNames is None:
            # --- Create constant multiplication value
            paramName = f"a_{self.info['injectionLocation']}"
            self.fitSynthetic["params"]["paramName"].append(paramName)
            params.add(paramName, value=1.0, min=0)

        # --- Only create the constant value for each bolometer if preforming a cross-calib
        else:
            for bolo_ in boloNames:
                paramName = f"{bolo_}"
                self.fitSynthetic["params"]["paramName"].append(paramName)
                params.add(paramName, value=0.3, min=0)

        # --- The exponential decay factor for each emission, only do one for helical directions
        for emissionName in self.info["emissionNames"]:

            paramName = None
            min = 0.0
            if "clockwise" in emissionName:
                min = 0.4
                paramName = f"b_clockwise_{self.info['injectionLocation']}"
                if paramName not in self.fitSynthetic["params"]["paramName"]:
                    self.fitSynthetic["params"]["paramName"].append(paramName)
            elif "counterClock" in emissionName:
                min = 0.4
                paramName = f"b_counterClock_{self.info['injectionLocation']}"
                if paramName not in self.fitSynthetic["params"]["paramName"]:
                    self.fitSynthetic["params"]["paramName"].append(paramName)
            else:
                paramName = f"b_{emissionName}_{self.info['injectionLocation']}"

            if paramName is not None:
                self.fitSynthetic["params"]["paramName"].append(paramName)
                params.add(
                    paramName, value=2.0, vary=varyScaleFactor, min=min, max=15.0
                )

        self.fitSynthetic["params"]["params"] = params
