# class_radiation.py
# This file contains SXR and PRAD signal classes which inherent parameters
# from the general signal class in class_generic.py

import os
import sys

# --- Path to where the Globals.py file is!
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import h5py
from mdsthin import MDSplus
import numpy as np
import scipy.constants
from scipy.signal import savgol_filter
from Util_SXR import filter_data
from main.Globals import *
from main.Util import read_h5


class SXRSignal:
    """
    Basic SXRSignal class.

        arrays:
            SX45F, SX90PF, SX90MF, DISRADU
    """

    def __init__(
        self,
        shotNumber=0,
        arrayName="",
        offsetWindow=None,
        filterWindowLength=21,
        filterPolyOrder=3,
        filterType="hanning",
        badChannels=[],
        invertedChannels=[],
        truncateTimeRange=[],
        runner=True,
    ):

        self.info = {}
        self.info["shotNumber"] = shotNumber
        self.info["arrayName"] = arrayName
        self.info["filterWindowLength"] = filterWindowLength
        self.info["filterPolyOrder"] = filterPolyOrder
        self.info["filterType"] = filterType
        self.info["badChannels"] = badChannels
        self.info["invertedChannels"] = invertedChannels
        self.info["offsetWindow"] = offsetWindow
        self.info["truncateTimeRange"] = truncateTimeRange

        SUPPORTED_arrayNameS = ["SX90PF", "SX90MF", "DISRADU", "SX45F"]
        if arrayName not in SUPPORTED_arrayNameS:
            print(f"ERROR!, the {arrayName} is not supported")
            print(f"please enter one of these values: {SUPPORTED_arrayNameS}")

        else:
            if runner:
                self._process_signal()

    def _process_signal(self):
        """
        This definition will call:
        1) grab_data
        2) correct the offset
        3) filter the data
        4) Truncate the filtered data
        5) Apply the calibration

        """
        self._num_channels()
        self._grab_data()
        self._filter_data()
        # self._correct_inverted_channels()
        self._correct_offset(offsetWindow=self.info["offsetWindow"])
        self._zero_negative_values()
        self.truncate_data(timeWindow=self.info["truncateTimeRange"])
        if len(self.info["badChannels"]) > 0:
            self._zero_bad_channels()

        # self.calibrate_data()

    def _open_connection(self):
        """
        Opens a connection to Atlas
        """
        try:
            self.conn = MDSplus.Connection("atlas.gat.com")
            self.conn.openTree("SPECTROSCOPY", self.info["shotNumber"])
        except Exception:
            self.conn = None

    def _close_connection(self):
        """
        Closes the connection to the tree
        """
        try:
            if self.conn:
                self.conn.closeTree("SPECTROSCOPY", self.info["shotNumber"])
        except Exception:
            pass

    def _num_channels(self):
        if self.info["arrayName"] in ["SX90PF", "SX90MF"]:
            self.info["numChannels"] = 32
        elif self.info["arrayName"] == "DISRADU":
            self.info["numChannels"] = 30
        elif self.info["arrayName"] == "SX45F":
            self.info["numChannels"] = 12

    def _grab_data(self):
        """
        Grabs the data from the tree
        """

        self._open_connection()
        arrayName = self.info["arrayName"]
        shotNumber = self.info["shotNumber"]

        self.data = {}
        self.data["channelOrder"] = []

        for ii in range(1, self.info["numChannels"] + 1):
            tag_ = arrayName + "0" * int(ii < 10) + str(ii)
            self.data["channelOrder"].append(tag_)
            print(f"Fetching data for {tag_}")
            try:
                # --- Grab the data
                try:
                    temp_ = self.conn.get("_sig= \\" + arrayName + ":" + tag_).data()  # type: ignore
                except:
                    temp_ = self.conn.get(  # type: ignore
                        ('_sig=ptdata2("' + tag_ + '",' + str(shotNumber) + ")")
                    ).data()

                temp_ = np.array(temp_)

                # --- Try and grab the time data, only once though!
                if "TIME_RAW" not in self.data:
                    try:
                        temp_t = self.conn.get("dim_of(_sig)").data()  # type: ignore
                        temp_t = np.array(temp_t)
                        self.data["TIME_RAW"] = temp_t.copy()
                    except Exception as e:
                        print(f"Error! Could not grab time data {e}")

                # --- Store the data, I don't know why we need to add +1 to the
                # end range when you don't need to for time.
                if "DATA" not in self.data:
                    self.data["DATA"] = np.zeros(
                        (self.info["numChannels"], temp_.shape[0])
                    )
                self.data["DATA"][ii - 1, :] = temp_

            except Exception as e:
                print(f"Error while trying to grab {tag_}, {e}")
                self._close_connection()

        self._close_connection()

    def _zero_bad_channels(
        self,
    ) -> None:
        """Definition to find the bad channels and NaN the signal out"""

        # Subtract 1 from each value in bad chan to match array indicies
        bad = []
        if len(self.info["badChannels"]) > 0:
            [bad.append(ii - 1) for ii in self.info["badChannels"]]

        for ii in bad:
            self.data["DATA"][ii, :] = 0.0
            if "PROCESSED_SIGNAL" in self.data:
                self.data["PROCESSED_SIGNAL"][ii, :] = 0
                self.data["PROCESSED_SIGNAL_SIGMA"][ii, :] = 0
            print(
                f"Signals set to zero for bad channel: {self.data['channelOrder'][ii]}"
            )

    def _correct_offset(self, offsetWindow=[]):
        """
        Definition to correct the offset
        """

        if len(offsetWindow) == 0:
            # Subtract from the last 200 ms of the signal
            endT = self.get_time()[-1]
            offsetWindow = [endT - 200, endT]

        start = self._get_time_loc(offsetWindow[0])
        end = self._get_time_loc(offsetWindow[1])

        temp = np.zeros(self.data["DATA"].shape)

        for ii in range(self.get_raw_data().shape[0]):
            offset = np.mean(self.get_observed()[ii, start:end])
            temp[ii, :] = self.get_observed()[ii, :] - offset

        self.data["PROCESSED_SIGNAL"] = temp.copy()

    def _zero_negative_values(self):
        """
        Definition to set negative values to zero
        """

        for ii in range(0, self.get_raw_data().shape[0]):
            temp = self.get_observed()[ii, :]
            indx_ = temp <= 1.0e-6
            temp[indx_] = 0.0

            # --- Set the error to 10 mV where the signal is negative
            temp_e = self.data["PROCESSED_SIGNAL_SIGMA"][ii, :]
            temp_e[indx_] = 0.01

            self.data["PROCESSED_SIGNAL"][ii, :] = temp.copy()
            # self.data["PROCESSED_SIGNAL_SIGMA"][ii,:] = temp_e.copy()

    def _correct_inverted_channels(self):
        """Definition to find the bad channels and NaN the signal out"""

        bad = []
        if len(self.info["invertedChannels"]) > 0:
            [bad.append(ii - 1) for ii in self.info["badChannels"]]

        for ii in bad:
            self.data["DATA"][ii, :] = 0.0
            if "PROCESSED_SIGNAL" in self.data:
                self.data["PROCESSED_SIGNAL"][ii, :] = 0
                self.data["PROCESSED_SIGNAL_SIGMA"][ii, :] = 0
            print(f"Channel: {self.data['channelOrder'][ii]} inverted")

    def _get_time_loc(self, time):
        return np.abs(self.get_time() - time).argmin().item()

    def get_time(self, raw=False):
        if raw:
            return self.data["TIME_RAW"]
        elif "TIME" not in self.data:
            return self.data["TIME_RAW"]
        else:
            return self.data["TIME"]

    def get_raw_data(self, time=None):
        if time is None:
            return self.data["DATA"]
        else:
            loc = np.abs(self.get_time() - time).argmin()
            return self.data["DATA"][:, loc]

    def get_observed(self, time=None):
        if time is None:
            if "PROCESSED_SIGNAL" in self.data:
                return self.data["PROCESSED_SIGNAL"]
            else:
                return self.data["DATA"]
        else:
            loc = np.abs(self.get_time() - time).argmin()
            if "PROCESSED_SIGNAL" in self.data:
                return self.data["PROCESSED_SIGNAL"][:, loc]
            else:
                return self.data[:, loc]

    def get_calibrated(self):
        return self.data["DATA_CALIBRATED"]

    def _filter_data_savgol(self):
        """This definition uses a savgol filter to filter the data"""

        numChannels = self.info["numChannels"]
        windowLength = self.info["filterWindowLength"]
        filterPolyOrder = self.info["filterPolyOrder"]

        if "PROCESSED_SIGNAL" not in self.data:
            self.data["PROCESSED_SIGNAL"] = np.zeros(self.get_raw_data().shape)

        for ii in range(0, numChannels):
            temp = savgol_filter(
                self.get_raw_data()[ii, :],
                windowLength,
                filterPolyOrder,
            )

            self.data["PROCESSED_SIGNAL"][ii, :] = temp.copy()

    def _filter_data(self):
        """This definition will filter the data using a hanning window"""
        if self.info is None:
            return

        if "PROCESSED_SIGNAL" not in self.data:
            self.data["PROCESSED_SIGNAL"] = np.zeros(self.get_raw_data().shape)
            self.data["PROCESSED_SIGNAL_SIGMA"] = np.zeros(self.get_raw_data().shape)

        for ii in range(0, self.info["numChannels"]):
            d, d_e = filter_data(
                data=self.get_raw_data()[ii, :],
                filter_type=self.info["filterType"],
                window_len=self.info["filterWindowLength"],
            )

            self.data["PROCESSED_SIGNAL"][ii, :] = d.copy()
            self.data["PROCESSED_SIGNAL_SIGMA"][ii, :] = d_e.copy()

    def truncate_data(self, timeWindow=[]):
        """This definition will truncate the data within the structure"""
        if len(timeWindow) > 0:

            start = self._get_time_loc(timeWindow[0])
            end = self._get_time_loc(timeWindow[1])
            temp_data = np.zeros((self.get_observed().shape[0], end - start))
            temp_data_e = np.zeros((self.get_observed().shape[0], end - start))

            for ii in range(0, self.get_observed().shape[0]):
                temp_data[ii, :] = self.get_observed()[ii, start:end]
                temp_data_e[ii, :] = self.data["PROCESSED_SIGNAL_SIGMA"][ii, start:end]

            self.data["PROCESSED_SIGNAL"] = temp_data.copy()
            self.data["PROCESSED_SIGNAL_SIGMA"] = temp_data.copy()
            self.data["TIME"] = self.get_time(raw=True).copy()[start:end]

    def _get_calib_info(self, path_calib="./sxrCalibs/"):
        """This definition will read the SXRsettingsPA or SXRsettingsTA.dat to grab the
        calibration self. A lot of this code was taken from the pytomo load_SXR.py routine
        """

        try:
            open_file = False
            P = False
            PA = False
            pinhole_diameter = [0]
            pinhole_thickness = [0]
            if self.info["shotNumber"] < 156200:
                raise Exception("No filter information for shot prior to 156200")

            # Filter widths for specific shot ranges

            # Figure out what file to grab
            if self.info["arrayName"].upper() in ["SX90PF", "SX90MF"]:
                path_calib += "SXRsettingsPA.dat"
                PA = self.info["arrayName"].upper() in ["SX90PF", "SX90MF"]
                P = self.info["arrayName"].upper() in ["SX90PF"]
                open_file = True
                pinhole_diameter = [0.0, 200.0, 1070.0, 1.95e3, 400.0, 1.95e3]
                pinhole_thickness = [0.0, 50.0, 25.0, 1.3, 50.0, 25.0]
                self.info["ELEMENT_AREA"] = 2.0 * 5.0 / 1.0e6
                self.info["CENTER_TO_PINHOLE"] = 3.0  # 2.95
                self.info["CENTER_CENTER_SPACING"] = 0.212
                self.info["nchan"] = 16
            elif self.info["arrayName"].upper() == "SX45F":
                path_calib += "SXRsettings45U.dat"
                PA = False
                P = False
                open_file = True
                pinhole_diameter = [0.0, 3.6e3, 3.6e3, 3.6e3, 3.6e3, 3.6e3]
                pinhole_thickness = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                self.info["ELEMENT_AREA"] = 4.1 * 0.75 / 1.0e6
                self.info["CENTER_TO_PINHOLE"] = 2.7
                self.info["CENTER_CENTER_SPACING"] = 0.095
                self.info["nchan"] = 20
            elif self.info["arrayName"].upper() == "DISRADU":
                self.info["Rc"] = 2.0e3
                self.info["GAIN"] = 1.0
                self.info["PINHOLE_DIAMETER"] = 200.0
                self.info["PINHOLE_THICKNESS"] = 50.0
                self.info["ELEMENT_AREA"] = 2.0 * 5.0 / 1.0e6
                self.info["CENTER_TO_PINHOLE"] = 2.95
                self.info["CENTER_CENTER_SPACING"] = 0.212
                self.info["nchan"] = 16
                open_file = False

            if open_file:
                # Read the file
                file_data = open(path_calib, "r")

                # Create blank arrays
                shots = []
                Rc = []
                Gain = []
                Filt = []

                read_file = False
                for line in file_data:
                    if read_file:
                        line = line.split()
                        shots.append(int(line[0]))

                        Rc.append(float(line[1 + P].replace("k", "e3")))
                        Gain.append(float(line[2 + PA + P]))
                        Filt.append(int(line[3 + PA * 2 + P]))

                    # Start reading the file once you hit the shots
                    if line[:4] == "shot":
                        read_file = True

                # Close the file
                file_data.close()

                # Find where the shot is located
                try:
                    loc = np.where(np.array(shots) >= int(self.info["SHOT"]))[0][0] - 1
                # Exception to use the last shot
                except Exception:
                    loc = len(shots) - 1

                self.info["Rc"] = Rc[loc]
                self.info["GAIN"] = Gain[loc]
                self.info["FILTER_NUMBER"] = Filt[loc]
                self.info["PINHOLE_DIAMETER"] = pinhole_diameter[Filt[loc]]
                self.info["PINHOLE_THICKNESS"] = pinhole_thickness[Filt[loc]]

        except Exception as e:
            print("Progam failed while trying to find SXR calibration file")
            print("Error: {}".format(e))

    def calibrate_data(self, effective_response=0.12):
        """
        This definition will correct the data due to different entudues. This code was
        assembled from Eric Hollmann's matlab sxr_calib.m and the pytomo load_SXR.py code

        OUTPUT:
               self.self.data['DATA_CALIBRATED'] = Calibrated SXR data, if the data
                                                   exists in self.data['DATA_PROCESSED']
                                                   prior to running this program.

               self.info['ENTENDUE'] = Etendue correction factors for this array

               self.info['CALIBRATION_FACTOR'] = Calibration factors for each channel

               self.info['CALIBRATION_FACTOR_UNITS'] = Calibration factor units
        """

        # Call the routine to grab the information for the shot
        self._get_calib_info()

        # measured for filter 5 (high Te)
        if self.info["arrayName"] == "SX90MF" and self.info["FILTER_NUMBER"] == 3:
            if True:
                self.info["ENTENDUE"] = (
                    np.array(
                        [
                            1.749,
                            2.18,
                            2.442,
                            2.646,
                            2.754,
                            2.558,
                            2.592,
                            2.783,
                            2.824,
                            2.833,
                            2.783,
                            2.646,
                            2.504,
                            2.309,
                            2.031,
                            1.765,
                            2.347,
                            2.646,
                            2.895,
                            3.098,
                            3.244,
                            3.352,
                            3.414,
                            3.447,
                            3.447,
                            3.352,
                            3.211,
                            3.053,
                            2.866,
                            2.625,
                            2.23,
                            1.661,
                        ]
                    )
                    * 1e-8
                )
        elif self.info["arrayName"] == "SX90MF" and self.info["FILTER_NUMBER"] == 5:
            if True:
                self.info["ENTENDUE"] = (
                    np.array(
                        [
                            2.141,
                            2.338,
                            2.528,
                            2.702,
                            2.882,
                            2.695,
                            2.728,
                            2.985,
                            2.989,
                            2.971,
                            2.858,
                            2.711,
                            2.538,
                            2.349,
                            2.152,
                            1.900,
                            2.074,
                            2.331,
                            2.589,
                            2.836,
                            3.056,
                            3.234,
                            3.356,
                            3.41,
                            3.382,
                            3.282,
                            3.122,
                            2.914,
                            2.675,
                            2.419,
                            2.161,
                            1.91,
                        ]
                    )
                    * 1e-8
                )
                # self.info['ENTENDUE'] *= (self.info['PINHOLE_DIAMETER']**2
                #                          / 1.95e3**2)

        elif self.info["arrayName"] == "SX90PF" and self.info["FILTER_NUMBER"] == 3:
            if True:
                self.info["ENTENDUE"] = (
                    np.array(
                        [
                            1.628,
                            2.068,
                            2.356,
                            2.546,
                            2.648,
                            2.607,
                            2.713,
                            2.847,
                            2.945,
                            2.859,
                            2.758,
                            2.632,
                            2.498,
                            2.315,
                            2.067,
                            1.645,
                            1.877,
                            2.076,
                            2.292,
                            2.479,
                            2.658,
                            2.766,
                            2.865,
                            2.931,
                            2.946,
                            2.905,
                            2.803,
                            2.654,
                            2.492,
                            2.23,
                            1.873,
                            1.279,
                        ]
                    )
                    * 1e-8
                )
        elif self.info["arrayName"] == "SX90PF" and self.info["FILTER_NUMBER"] == 5:
            if True:
                self.info["ENTENDUE"] = (
                    np.array(
                        [
                            1.845,
                            2.073,
                            2.302,
                            2.523,
                            2.72,
                            2.671,
                            2.813,
                            3.015,
                            3.031,
                            2.948,
                            2.811,
                            2.631,
                            2.421,
                            2.195,
                            1.965,
                            1.74,
                            1.881,
                            2.182,
                            2.425,
                            2.663,
                            2.882,
                            3.069,
                            3.211,
                            3.296,
                            3.314,
                            3.259,
                            3.142,
                            2.975,
                            2.769,
                            2.539,
                            2.297,
                            2.055,
                        ]
                    )
                    * 1e-8
                )
                # self.info['ENTENDUE'] *= (self.info['PINHOLE_DIAMETER']**2
                #                          / 1.95e3**2)

        else:
            ap_area = (
                0.25 * scipy.constants.pi * (self.info["PINHOLE_DIAMETER"] / 1.0e4) ** 2
            )  # aperture area [cm^2]

            dist_from_ctr = (
                np.arange(-int(self.info["nchan"] / 2), int(self.info["nchan"] / 2))
                + 0.5
            ) * self.info[
                "CENTER_CENTER_SPACING"
            ]  # [cm]

            tanpsi = dist_from_ctr / self.info["CENTER_TO_PINHOLE"]

            cos4 = (tanpsi**2 + 1.0) ** (-2.0)

            thick_factor = (
                abs(tanpsi)
                * (-4.0 / scipy.constants.pi)
                * (self.info["PINHOLE_THICKNESS"] / self.info["PINHOLE_DIAMETER"])
                + 1.0
            )

            entendue = (
                cos4
                * thick_factor
                * self.info["ELEMENT_AREA"]
                * ap_area
                / self.info["CENTER_TO_PINHOLE"] ** 2
                * 1.0e2
            )  # units m2 sr

            if self.info["arrayName"] in ["SX90MF", "SX90PF", "DISRADU"]:
                self.info["ENTENDUE"] = np.concatenate([entendue, np.flip(entendue)])
            else:
                # from Eric Hollmann, in load_SXR.py in pytomo, for SX45F
                active = [18, 17, 15, 13, 11, 10, 9, 8, 7, 6, 5, 3]
                self.info["ENTENDUE"] = entendue[active]

        self.info["CALIBRATION_FACTOR"] = (
            1.0
            / 0.5  # 50 Ohm termination divides by 2
            / (self.info["Rc"] * self.info["GAIN"])
            / effective_response
            / self.info["ENTENDUE"]
        )

        self.info["CALIBRATION_FACTOR_UNITS"] = "W / (m2 sr V)"

        # Apply the correction factor if the data exits
        if "PROCESSED_SIGNAL" in self.data:
            print("Calibrating data")
            self.data["DATA_CALIBRATED"] = self.data["PROCESSED_SIGNAL"].copy()
            self.data["DATA_CALIBRATED_ERROR"] = self.data[
                "PROCESSED_SIGNAL_SIGMA"
            ].copy()

            for ii in range(self.data["DATA_CALIBRATED"].shape[0]):
                loc_ = ii
                if self.info["arrayName"] == "DISRADU":
                    loc_ = ii + 1
                self.data["DATA_CALIBRATED"][ii, :] *= self.info["CALIBRATION_FACTOR"][
                    loc_
                ]
                self.data["DATA_CALIBRATED_ERROR"][ii, :] *= self.info[
                    "CALIBRATION_FACTOR"
                ][loc_]
            self.data["DATA_CALIBRATED_UNITS"] = "W / (m2 sr)"

    def save_data(self):
        """Saves the data"""
        print(EMIS3D_INPUTS_DIRECTORY)
        path = os.path.join(
            EMIS3D_INPUTS_DIRECTORY, "DIII-D", "sxrData", str(self.info["shotNumber"])
        )

        if not os.path.exists(path):
            os.makedirs(path)

        fileName = f"{self.info['shotNumber']}_{self.info['arrayName']}_PROCESSED.h5"
        pathFileName = os.path.join(path, fileName)

        f = h5py.File(pathFileName, "w")

        for key in [
            "channelOrder",
            "TIME",
            "DATA_CALIBRATED",
            "DATA_CALIBRATED_ERROR",
            "DATA_CALIBRATED_UNITS",
        ]:
            f.create_dataset(name=key, data=self.data[key])
        f.close()

        print(f"File saved to {pathFileName}")

    def load_data(self):
        """Loads the data"""
        path = os.path.join(
            EMIS3D_INPUTS_DIRECTORY, "DIII-D", "sxrData", str(self.info["shotNumber"])
        )

        fileName = f"{self.info['shotNumber']}_{self.info['arrayName']}_PROCESSED.h5"
        pathFileName = os.path.join(path, fileName)

        self.info["pathFileName"] = pathFileName

        if not os.path.exists(pathFileName):
            print(f"Error! File does not exist: {pathFileName}")
            return

        self.data = read_h5(pathFileName)

        print(f"File loaded from {pathFileName}")

    def integrate_signal(self, time_start=0, dt=0.1, numIntegrations=1):
        """This definition will integrate the signal between the start and end times"""

        if time_start == 0:
            time_start = self.get_time()[0]

        int_signal = np.zeros((self.get_calibrated().shape[0], numIntegrations))
        int_signal_error = np.zeros((self.get_calibrated().shape[0], numIntegrations))

        for n in range(numIntegrations):
            loc_start = self._get_time_loc(time_start)
            loc_end = self._get_time_loc(time_start + dt)

            for ii in range(self.get_calibrated().shape[0]):
                temp_ = self.get_calibrated()[ii, loc_start:loc_end]
                temp_e = self.data["DATA_CALIBRATED_ERROR"][ii, loc_start:loc_end]

                dt_ = np.diff(self.get_time()[loc_start:loc_end])
                dt_ = np.append(dt_, dt_[-1])

                int_signal[ii, n] = np.sum(temp_ * dt_).item()
                int_signal_error[ii, n] = np.sqrt(np.sum((temp_e * dt_) ** 2)).item()

            # --- Update the time_start
            time_start += dt

        return int_signal, int_signal_error
