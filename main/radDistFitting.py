# radDist_Fitting.py
"""
Helper class used to set up a loaded radDist for fitting, maps signals, loads radDist,
creates the parameters, etc.

Written by JLH Aug. 2025
"""

import numpy as np
from lmfit import Parameters
from main.Util import load_json, convert_arrays_to_list, find_max_nested_lists
from main.Util_emis3D import loc_tag


class RadDistFitting:
    """
    radDist class used when fitting each radDist. This class preforms actions
    such as loading the data, organizing it based of a specific channel order,
    generating the initial parameters to be used while fitting and more.
    """

    def __init__(self, radDistPath: str | None = None) -> None:
        self.info = {}
        self.info["radDistPath"] = radDistPath
        self._load_ok = False

        if radDistPath is not None:
            self._load_radDist()
            if self._load_ok:
                self._map_signals()

    def _load_radDist(self) -> None:
        """
        Loads the radDist given by radDistPath
        """
        try:
            temp = load_json(self.info["radDistPath"])
            self.data = temp["data"]
            self.info.update(temp["info"])
            self._load_ok = True

        except Exception as e:
            print(f"Could not load '{self.info['radDistPath']}': {e}")

    @staticmethod
    def _as_segment_list(value) -> list:
        """
        Normalize a per-channel entry to a list of toroidal segments.

        Older saved radDists stored a single float per channel (one toroidal
        observation plane); newer radDists store a list with one entry per
        toroidal segment of the observation. Scalars are promoted to
        single-element lists so downstream code only sees lists.
        """
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _map_signals(self) -> None:
        """
        Creates a dictionary that maps the channel name to the synthetic signal.

        Each channel maps to a list of values, one per toroidal observation
        segment (single-element lists for poloidal-fan cameras).
        """
        units = self.info["units"]
        bolos = list(self.data[units]["channelOrder"].keys())
        self.maps = {}
        self.data_maps = {}
        for emissionName in self.info["emissionNames"]:
            self.data_maps[emissionName] = {}
            for val in ["observed_phi_loc", "data", "data_error"]:
                self.data_maps[emissionName][val] = {}

        # --- The toroidal observation angles; formerly saved as "scaleFactor"
        phi_loc = self.data.get("observed_phi_loc", None)
        if phi_loc is None:
            raise KeyError("radDist data contains neither 'observed_phi_loc' key")

        # --- Loop over each signal
        for bolo_ in bolos:
            channels = self.data[units]["channelOrder"][bolo_]

            for emissionName in self.info["emissionNames"]:
                scale_ = self.data["scaleFactor"][emissionName][bolo_]
                map_scale = dict(zip(channels, scale_))

                data_ = self.data[units][emissionName][bolo_]
                phi_ = [self._as_segment_list(v) for v in phi_loc[emissionName][bolo_]]
                map_phi = dict(zip(channels, phi_))
                data_ = [
                    self._as_segment_list(v)
                    for v in self.data[units][emissionName][bolo_]
                ]

                map_data = dict(zip(channels, data_))

                data_error = [
                    self._as_segment_list(v)
                    for v in self.data[f"{units}_error"][emissionName][bolo_]
                ]
                map_data_error = dict(zip(channels, data_error))

                self.data_maps[emissionName]["data"].update(map_data)
                self.data_maps[emissionName]["data_error"].update(map_data_error)
                self.data_maps[emissionName]["observed_phi_loc"].update(map_phi)

    def prepare_for_fits(
        self, channelOrder: list, data_max: float | None = None
    ) -> None:
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
            for val in ["observed_phi_loc", "data", "data_error"]:
                self.fitSynthetic[emissionName][val] = []

            for bolo_ in channelOrder:
                temp_data = []
                temp_data_error = []
                temp_phi = []
                for chan in bolo_:

                    # --- Make sure the channel is in the radDist
                    if chan not in self.data_maps[emissionName]["data"]:
                        if "ERROR CHANNELS" not in self.info:
                            self.info["ERROR CHANNELS"] = ""
                        if chan not in self.info["ERROR CHANNELS"]:
                            self.info["ERROR CHANNELS"] += f"{chan}, "

                    # --- Each channel holds a list with one entry per toroidal
                    # observation segment; missing channels get a single zero
                    # segment so the shapes stay consistent.
                    temp_data.append(
                        self.data_maps[emissionName]["data"].get(chan, [0.0])
                    )
                    temp_data_error.append(
                        self.data_maps[emissionName]["data_error"].get(chan, [1.0e19])
                    )
                    # --- The toroidal location(s) of the channel's observation
                    temp_phi.append(
                        self.data_maps[emissionName]["observed_phi_loc"].get(
                            chan, [0.0]
                        )
                    )

                self.fitSynthetic[emissionName]["data"].append(temp_data)
                self.fitSynthetic[emissionName]["data_error"].append(temp_data_error)
                self.fitSynthetic[emissionName]["observed_phi_loc"].append(temp_phi)

        # --- Find the synthetic scaling factor, set to 1 if the user doesn't provide data_max
        scale = 1.0
        if data_max is not None:
            scale_ = []
            for emissionName in self.info["emissionNames"]:
                synth_max = find_max_nested_lists(
                    self.fitSynthetic[emissionName]["data"]
                )
                if synth_max is not None and synth_max > 0:
                    scale_.append(data_max / synth_max)

            # --- Use the max value
            if len(scale_) > 0:
                scale = np.nanmax(scale_)

        # --- Scale the synthetic data
        for emissionName in self.info["emissionNames"]:
            self.fitSynthetic[emissionName]["scaleSynth"] = scale
            # --- Scale the data. Channels may hold different numbers of
            # toroidal segments, so scale channel-by-channel.
            temp_ = []
            for val in self.fitSynthetic[emissionName]["data"]:
                temp_.append([scale * np.asarray(chan) for chan in val])
            self.fitSynthetic[emissionName]["data"] = convert_arrays_to_list(temp_)

            # --- Scale the error
            temp_ = []
            for val in self.fitSynthetic[emissionName]["data_error"]:
                temp_.append([scale * np.asarray(chan) for chan in val])
            self.fitSynthetic[emissionName]["data_error"] = convert_arrays_to_list(
                temp_
            )

    def create_parameters(
        self,
        boloNames: list | None = None,
        enable_dphi_scaling: bool = False,
        vary_peak_rad_location: bool = False,
        scale_def: str = "",
    ) -> None:
        """
        Creates the LMFIT parameters for the radDist
        """
        self.fitSynthetic["params"] = {}
        self.fitSynthetic["params"]["paramName"] = []
        params = Parameters()

        # --- Create parameters for the normal fitting case
        if boloNames is None:
            # --- Create constant multiplication value
            paramName = f"a_{loc_tag(self.info['injectionLocation'])}"
            self.fitSynthetic["params"]["paramName"].append(paramName)
            params.add(paramName, value=1.0, min=1.0, max=6.0)

        # --- Only create the constant value for each bolometer if preforming a cross-calib
        else:
            for bolo_ in boloNames:
                paramName = f"{bolo_}"
                self.fitSynthetic["params"]["paramName"].append(paramName)
                params.add(paramName, value=0.3, min=0)

        # --- Create a peak radiation variable if vary_peak_rad_location is True:
        if vary_peak_rad_location:
            default_ = np.deg2rad(int(self.info["injectionLocation"]))
            default_ = np.deg2rad(225.0)
            paramName = f"peak_rad_loc"
            self.fitSynthetic["params"]["paramName"].append(paramName)
            params.add(
                paramName,
                value=default_,  # Start at the injection location
                min=0.1,
                max=2.0 * np.pi,
                vary=True,
            )

        # --- The exponential decay factor for each emission, only do one for helical directions
        for emissionName in self.info["emissionNames"]:
            paramName = None
            min_ = 0.0
            max_ = 10.0
            paramName = f"b_{emissionName}_{loc_tag(self.info['injectionLocation'])}"

            if scale_def.upper() == "LINEAR":
                min_ = 0.01
                max_ = 0.4

            # --- Add it to the params pool
            if (
                paramName is not None
                and paramName not in self.fitSynthetic["params"]["paramName"]
            ):
                self.fitSynthetic["params"]["paramName"].append(paramName)
                params.add(
                    paramName,
                    value=(max_ - min_) / 2.0,
                    vary=enable_dphi_scaling,
                    min=min_,
                    max=max_,
                )

        self.fitSynthetic["params"]["params"] = params
