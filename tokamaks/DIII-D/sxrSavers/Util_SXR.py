# Util_SXR.py
"""
Definitions to grab, store, and filter SXR data on DIII-D

"""

from os.path import dirname, join, realpath

import h5py
import numpy as np
import scipy.constants

FILE_PATH = dirname(realpath(__file__))
PARENT_DIRECTORY = dirname(FILE_PATH)


def _get_calib_info(ShotNumber, ArrayName=None):
    """
    This definition will read the SXRsettingsPA or SXRsettingsTA.dat to grab the
    calibration data. A lot of this code was taken from the pytomo load_SXR.py routine
    """

    try:
        if ArrayName is None:
            raise Exception("ArrayName must be specified!")

        info = {}
        info["ARRAY"] = ArrayName.upper()
        if ShotNumber < 156200:
            raise Exception("No filter information for shot prior to 156200")

        # Filter widths for specific shot ranges

        # Figure out what file to grab
        if ArrayName.upper() in ["SX90RP1F", "SX90RM1F"]:
            path_calib = join(PARENT_DIRECTORY, "SXRsettingsPA.dat")
            PA = ArrayName.upper() in ["SX90RP1F", "SX90RM1F"]
            P = ArrayName.upper() in ["SX90RP1F"]
            open_file = True
            pinhole_diameter = [0.0, 200.0, 1070.0, 1.95e3, 400.0, 1.95e3]
            pinhole_thickness = [0.0, 50.0, 25.0, 1.3, 50.0, 25.0]
            info["ELEMENT_AREA"] = 2.0 * 5.0 / 1.0e6
            info["ELEMENT_WIDTH"] = 2.0 / 1.0e3
            info["ELEMENT_HEIGHT"] = 5.0 / 1.0e3
            info["CENTER_TO_PINHOLE"] = 3.0  # 2.95
            info["CENTER_CENTER_SPACING"] = 0.212
            info["nchan"] = 16
        elif ArrayName.upper() == "SXR45":
            path_calib = join(EMIS3D_SXR_CALIB_DIRECTORY, "SXRsettings45U.dat")
            PA = False
            P = False
            open_file = True
            pinhole_diameter = [0.0, 3.6e3, 3.6e3, 3.6e3, 3.6e3, 3.6e3]
            pinhole_thickness = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            info["ELEMENT_AREA"] = 4.1 * 0.75 / 1.0e6
            info["ELEMENT_WIDTH"] = 2.0 / 1.0e3
            info["ELEMENT_HEIGHT"] = 5.0 / 1.0e3
            info["CENTER_TO_PINHOLE"] = 2.7
            info["CENTER_CENTER_SPACING"] = 0.095
            info["nchan"] = 20
        elif ArrayName.upper() == "DISRADU":
            info["Rc"] = 2.0e3
            info["GAIN"] = 1.0
            info["PINHOLE_DIAMETER"] = 200.0
            info["PINHOLE_THICKNESS"] = 50.0
            info["ELEMENT_AREA"] = 2.0 * 5.0 / 1.0e6
            info["ELEMENT_WIDTH"] = 2.0 / 1.0e3
            info["ELEMENT_HEIGHT"] = 5.0 / 1.0e3
            info["CENTER_TO_PINHOLE"] = 2.95
            info["CENTER_CENTER_SPACING"] = 0.212
            info["FILTER_NUMBER"] = 10000
            info["nchan"] = 16
            open_file = False

        else:
            raise Exception("Not a valid array!")

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
                loc = np.where(np.array(shots) >= int(ShotNumber))[0][0] - 1
            # Exception to use the last shot
            except Exception:
                loc = len(shots) - 1

            info["Rc"] = Rc[loc]
            info["GAIN"] = Gain[loc]
            info["FILTER_NUMBER"] = Filt[loc]
            info["PINHOLE_DIAMETER"] = pinhole_diameter[Filt[loc]]
            info["PINHOLE_THICKNESS"] = pinhole_thickness[Filt[loc]]

        return info

    except Exception as e:
        print("Progam failed while trying to find SXR calibration file")
        print("Error: {}".format(e))


def get_calibration(ShotNumber, ArrayName=None, effective_response=0.12, etendue=None):
    """
    This definition will correct the data due to different entudues. This code was
    assembled from Eric Hollmann's matlab sxr_calib.m and the pytomo load_SXR.py code

    OUTPUT:
            data.data['DATA_CALIBRATED'] = Calibrated SXR data, if the data
                                                exists in data['DATA'] prior to
                                                running this program.
                                                Use grab_data() to grab the data.

            info['ETENDUE'] = Etendue correction factors for this array [m^2/sr]

            info['CALIBRATION_FACTOR'] = Calibration factors for each channel

            info['CALIBRATION_FACTOR_UNITS'] = Calibration factor units
    """

    # Call the routine to grab the information for the shot
    info = _get_calib_info(ShotNumber, ArrayName=ArrayName)

    info["dist_from_ctr"] = (
        np.arange(-int(info["nchan"] / 2), int(info["nchan"] / 2)) + 0.5
    ) * info["CENTER_CENTER_SPACING"]

    if etendue is None:
        if info["ARRAY"] in ["SX90RM1F", "SX90RP1F"]:
            # Measured for filter 5 (high Te)
            if info["ARRAY"] == "SX90RM1F":
                info["ETENDUE"] = np.array(
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

            if info["ARRAY"] == "SX90RP1F":
                info["ETENDUE"] = np.array(
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

            if info["FILTER_NUMBER"] == 3:
                if info["ARRAY"] == "SX90RM1F":
                    info["ETENDUE"] = np.array(
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

                if info["ARRAY"] == "SX90RP1F":
                    info["ETENDUE"] = np.array(
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
            else:
                info["ETENDUE"] *= info["PINHOLE_DIAMETER"] ** 2 / 1.95e3**2

            info["ETENDUE"] *= 1.0e-8

        else:
            ap_area = (
                0.25 * scipy.constants.pi * (info["PINHOLE_DIAMETER"] / 1.0e6) ** 2
            )

            tanpsi = info["dist_from_ctr"] / info["CENTER_TO_PINHOLE"]

            cos4 = (tanpsi**2 + 1.0) ** (-2.0)

            thick_factor = (
                abs(tanpsi)
                * (-4.0 / scipy.constants.pi)
                * (info["PINHOLE_THICKNESS"] / info["PINHOLE_DIAMETER"])
                + 1.0
            )

            ETENDUE = (
                cos4
                * thick_factor
                * info["ELEMENT_AREA"]
                * ap_area
                / info["CENTER_TO_PINHOLE"] ** 2
                * 1.0e4
            )

            if info["ARRAY"] in ["SX90RM1F", "SX90RP1F", "DISRADU"]:
                info["ETENDUE"] = np.concatenate([ETENDUE, np.flip(ETENDUE)])
            else:
                # from Eric Hollmann, in load_SXR.py in pytomo
                active = [18, 17, 15, 13, 11, 10, 9, 8, 7, 6, 5, 3]
                info["ETENDUE"] = ETENDUE[active]

    else:
        info["ETENDUE"] = np.array(etendue) * 4.0 * np.pi

    info["CALIBRATION_FACTOR"] = (
        4.0
        * scipy.constants.pi
        / 0.5  # 50 Ohm termination divides by 2
        / (info["Rc"] * info["GAIN"])
        / effective_response
        / info["ETENDUE"]
    )

    info["CALIBRATION_FACTOR_UNITS"] = "W / (m2 V)"

    return info


def scaling_factors(ShotNumber):
    """
    Returns the scaling factors to be applied to each array after calibration.

    This is done so each array will give the same output if they observe the same source.

    This is done independently of this program by comparing the CQ radiation of reconstructed
    power traces for each array over a whole day.
    """

    data = {}

    if ShotNumber > 170693 and ShotNumber < 170711:
        data["SXR45F"] = 0.0751
        data["SX90RM1F"] = 0.339
        data["SX90RP1F"] = 0.268
        data["DISRADU"] = 1.0

    elif ShotNumber == 0:  # > 184406 and ShotNumber < 184422:
        data["SXR45F"] = 1159.0
        data["SX90RM1F"] = 1.999
        data["SX90RP1F"] = 2.129
        data["DISRADU"] = 1.0
    elif ShotNumber == 184407:  # > 184406 and ShotNumber < 184422:
        data["SXR45F"] = 1.0
        data["SX90RM1F"] = 1.0
        data["SX90RP1F"] = 1.0
        data["DISRADU"] = 4.92
    else:
        data["SXR45F"] = 1.0
        data["SX90RM1F"] = 1.0
        data["SX90RP1F"] = 1.0
        data["DISRADU"] = 1.0

    return data


def read_h5(path):
    data = {}
    with h5py.File(path, "r") as f:
        for dset in traverse_datasets(f):
            cols = dset.strip().split("/")[1:]
            ensure_path(data=data, path=cols, default=f[dset][()])
    return data


def traverse_datasets(hdf_file):
    # Taken from: https://stackoverflow.com/questions/51548551/reading-nested-h5-group-into-numpy-array
    def h5py_dataset_iterator(g, prefix=""):
        for key in g.keys():
            item = g[key]
            path = f"{prefix}/{key}"
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path


def ensure_path(data, path, default=None, default_func=lambda x: x):
    """
    # Taken from: https://stackoverflow.com/questions/16333296/how-do-you-create-nested-dict-in-python
    Function:

    - Ensures a path exists within a nested dictionary

    Requires:

    - `data`:
        - Type: dict
        - What: A dictionary to check if the path exists
    - `path`:
        - Type: list of strs
        - What: The path to check

    Optional:

    - `default`:
        - Type: any
        - What: The default item to add to a path that does not yet exist
        - Default: None

    - `default_func`:
        - Type: function
        - What: A single input function that takes in the current path item (or default) and adjusts it
        - Default: `lambda x: x` # Returns the value in the dict or the default value if none was present
    """
    if len(path) > 1:
        if path[0] not in data:
            data[path[0]] = {}
        data[path[0]] = ensure_path(
            data=data[path[0]],
            path=path[1:],
            default=default,
            default_func=default_func,
        )
    else:
        if path[0] not in data:
            data[path[0]] = default
        data[path[0]] = default_func(data[path[0]])
    return data


def filter_data(data, filter_type="hanning", window_len=21):
    """The majority of the definition was taken from the scipycookbook.
       This definition will filter the VB data based on the following definitions:

    From numpy ::
            data        :: The data to filter, format [data].
            window_len  :: The window length for the specific filtering scheme
            run_ave     :: A running average, default 100 data points
            hanning     :: DEFAULT
            hamming     ::
            bartlett    ::
            blackman    ::


    """
    
    try:
        # print("filtering data using " + filter_type)

        temp_data = np.r_[
            data[window_len - 1 : 0 : -1], data[:], data[-2 : -window_len - 1 : -1]
        ]

        if filter_type == "run_ave":
            w = np.ones(window_len, "d")
        else:
            w = eval("np." + filter_type + "(window_len)")

        temp_data2 = np.convolve(w / w.sum(), temp_data, mode="valid")

        data_filtered =  temp_data2[
            int((window_len - 1) / 2) : int(
                temp_data2.shape[0] - ((window_len - 1) / 2)
            )
        ]

        # --- Finding the error
        delta_x = (
            data
            - temp_data2[
                int((window_len - 1) / 2) : int(
                    temp_data2.shape[0] - ((window_len - 1) / 2)
                )
            ]
        )
        temp_err = np.r_[
            delta_x[window_len - 1 : 0 : -1],
            delta_x[:],
            delta_x[-2 : -window_len - 1 : -1],
        ]
        temp_err2 = np.sqrt(np.convolve(w / w.sum(), temp_err**2, mode="valid"))

        data_filtered_error = temp_err2[
                int((window_len - 1) / 2) : int(
                    temp_err2.shape[0] - ((window_len - 1) / 2)
                )
            ]
        return data_filtered, data_filtered_error
    
    except:
        print("The signal couldn't be filtered! Skipping def filter_data.")
        return [], []




        
