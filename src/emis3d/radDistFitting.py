# radDist_Fitting.py
"""
Helper class used to set up a loaded radDist for fitting, maps signals, loads radDist,
creates the parameters, etc.

Written by JLH Aug. 2025
"""

import numpy as np
from lmfit import Parameters
from emis3d.util import load_json, convert_arrays_to_list
from emis3d.util_emis3D import loc_tag
from emis3d.globals import RADDIST_FORMAT


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
        self.load_error: str | None = None

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
            version = temp["info"].get("radDistFormat")
            if version != RADDIST_FORMAT:
                raise ValueError(f"saved in radDist format {version}, but this Emis3D reads format " f"{RADDIST_FORMAT}; regenerate it with make_radDists.py")
            self.data = temp["data"]
            self.info.update(temp["info"])
            self._load_ok = True

        except (OSError, ValueError, KeyError, TypeError) as e:
            # An unreadable or malformed file; the caller decides whether this is
            # fatal (see Emis3D._load_radDists). Other errors are bugs and propagate.
            self.load_error = f"{type(e).__name__}: {e}"

    @staticmethod
    def _as_segment_list(value) -> list:
        """A channel entry: the list of its toroidal segments."""
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"radDist channel entry {value!r} is not a list of toroidal segments; " "regenerate the radDist with the current Emis3D")
        return list(value)

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

        # --- The toroidal observation angles
        phi_loc = self.data["observed_phi_loc"]

        # --- Loop over each signal
        for bolo_ in bolos:
            channels = self.data[units]["channelOrder"][bolo_]

            for emissionName in self.info["emissionNames"]:
                data_ = self.data[units][emissionName][bolo_]
                phi_ = [self._as_segment_list(v) for v in phi_loc[emissionName][bolo_]]
                map_phi = dict(zip(channels, phi_))
                data_ = [self._as_segment_list(v) for v in self.data[units][emissionName][bolo_]]

                map_data = dict(zip(channels, data_))

                data_error = [self._as_segment_list(v) for v in self.data[f"{units}_error"][emissionName][bolo_]]
                map_data_error = dict(zip(channels, data_error))

                self.data_maps[emissionName]["data"].update(map_data)
                self.data_maps[emissionName]["data_error"].update(map_data_error)
                self.data_maps[emissionName]["observed_phi_loc"].update(map_phi)

    def prepare_for_fits(self, channelOrder: list, data_max: float | None = None) -> None:
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
                    temp_data.append(self.data_maps[emissionName]["data"].get(chan, [0.0]))
                    temp_data_error.append(self.data_maps[emissionName]["data_error"].get(chan, [1.0e19]))
                    # --- The toroidal location(s) of the channel's observation
                    temp_phi.append(self.data_maps[emissionName]["observed_phi_loc"].get(chan, [0.0]))

                self.fitSynthetic[emissionName]["data"].append(temp_data)
                self.fitSynthetic[emissionName]["data_error"].append(temp_data_error)
                self.fitSynthetic[emissionName]["observed_phi_loc"].append(temp_phi)

        # --- Normalization: peak of the summed model equals the data peak at a = 1
        scale = 1.0
        if data_max is not None:
            total = None
            for emissionName in self.info["emissionNames"]:
                per_chan = [[float(np.sum(seg)) for seg in group] for group in self.fitSynthetic[emissionName]["data"]]
                flat = np.concatenate([np.asarray(g, dtype=float) for g in per_chan])
                total = flat if total is None else total + flat
            total_max = np.nanmax(total) if total is not None and total.size else 0.0
            if total_max > 0.0:
                scale = data_max / total_max

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
            self.fitSynthetic[emissionName]["data_error"] = convert_arrays_to_list(temp_)

    def create_parameters(
        self,
        enable_dphi_scaling: bool = False,
        vary_peak_rad_location: bool = False,
        scale_def: str = "",
        peak_shift_max_degrees: float = 180.0,
        a_bounds: tuple | None = None,
        b_bounds: tuple | None = None,
    ) -> None:
        """
        Creates the LMFIT parameters for the radDist. peak_shift_max_degrees
        limits how far a fitted peak may move from the injector (run config).
        a_bounds, b_bounds: bounds of the amplitude and the toroidal decay
        (the run config's `bounds`; defaults in emis3d.globals).
        """
        from emis3d.globals import AMPLITUDE_BOUNDS, DECAY_BOUNDS

        a_lo, a_hi = a_bounds if a_bounds is not None else AMPLITUDE_BOUNDS
        b_default = DECAY_BOUNDS.get(scale_def.lower(), DECAY_BOUNDS["exponential"])
        b_lo, b_hi = b_bounds if b_bounds is not None else b_default
        self.fitSynthetic["params"] = {}
        self.fitSynthetic["params"]["paramName"] = []
        params = Parameters()

        # --- Constant multiplication value
        paramName = f"a_{loc_tag(self.info['injectionLocation'])}"
        self.fitSynthetic["params"]["paramName"].append(paramName)
        # Start at 1 (model peak = data peak); allow scaling down for
        # multi-component radDists and up for strong toroidal decay
        params.add(paramName, value=float(np.clip(1.0, a_lo, a_hi)), min=a_lo, max=a_hi)

        # --- Create a peak radiation variable if vary_peak_rad_location is True:
        if vary_peak_rad_location:
            two_pi = 2.0 * np.pi
            mu0 = np.deg2rad(float(self.info["injectionLocation"])) % two_pi
            window = np.deg2rad(float(peak_shift_max_degrees))
            params.add("injector_phi", value=mu0, vary=False)
            params.add("peak_shift", value=0.0, min=-window, max=window)
            params.add("peak_rad_loc", expr="(injector_phi + peak_shift) % (2*pi)")
            self.fitSynthetic["params"]["paramName"] += ["peak_shift", "peak_rad_loc"]

        # --- The exponential decay factor for each emission, only do one for helical directions
        for emissionName in self.info["emissionNames"]:
            min_, max_ = b_lo, b_hi
            paramName = f"b_{emissionName}_{loc_tag(self.info['injectionLocation'])}"

            # --- Add it to the params pool
            if paramName is not None and paramName not in self.fitSynthetic["params"]["paramName"]:
                self.fitSynthetic["params"]["paramName"].append(paramName)
                params.add(
                    paramName,
                    value=0.5 * (max_ + min_),
                    vary=enable_dphi_scaling,
                    min=min_,
                    max=max_,
                )

        self.fitSynthetic["params"]["params"] = params
