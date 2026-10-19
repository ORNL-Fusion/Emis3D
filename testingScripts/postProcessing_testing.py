# postProcessing_testing.py
"""
Tests the post-processing TPF calculation

"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from emis3d.Emis3D import Emis3D
import emis3d.Util_emis3D as Util_emis3D
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# NOTE: Program assumes you already have a fit done, and loads it

evalTime = 50.949
tokamakName = "JET"
runConfigName = "95709/95709_runConfig.yaml"
verbose = True

t = Emis3D(
    tokamakName=tokamakName,
    runConfigName=runConfigName,
    verbose=verbose,
    initialize=True,
)

t._load_bestFits(path="/Users/plh/Documents/git/Emis3D/inputs/JET/runs/95709/95709_bestFits_50.9490.dill")

'''
# --- Calculates the radiation amplitude distribution from the best fit

Main goals after a run:
1. Calculate overall radiation at each toroidal angle
2. Determine TPF

To do this:
1. Re-build "scaling function" to multiply the radDist power per bin for each emission
2. Add each emission together


'''

def _bestFit_radDist_parameters(emissionName) -> tuple[np.ndarray, np.ndarray]:
    """Finds the starting and ending phi for a given radDist"""
    phi = t.bestFits[evalTime]['radDist'].data['toroidalRadiatedPower'][emissionName]['phi_array']
    P_pol = t.bestFits[evalTime]['radDist'].data['toroidalRadiatedPower'][emissionName]['P_pol']
    return np.array(phi), np.array(P_pol)

def _bestFit_radDist_scaling(emissionName) -> float:
    """Returns the pre-fit synthetic / data normalization factor"""
    norm_factor= t.bestFits[evalTime]['synthetic_dict'][emissionName]['scaleSynth']
    return norm_factor

if t.info is not None:

    # --- First definition:
    # 1. Calculate the toroidal scaling factor for each emissionName based on the best fit parameters
    """Calculates the radiation amplitude distribution from the best fit"""

    # --- Create an empty dictionary
    t.bestFits[evalTime]["radiation_distribution"] = {}
    rad_distribution = t.bestFits[evalTime]["radiation_distribution"]
    emissionNames = t.bestFits[evalTime]["synthetic_dict"]["emissionNames"]

    scale_def = t.info["scale_def"]
    params = t.bestFits[evalTime]["fit"].params.valuesdict()

    # --- Check to see if the peak radiation location can be varied
    mu = t.bestFits[evalTime]["synthetic_dict"]["injectionLocation_rad"]
    if "peak_rad_loc" in params:
        mu = float(params["peak_rad_loc"])

    # Value at the end of each tag in params
    inj_loc_tag = Util_emis3D.loc_tag(t.bestFits[evalTime]["synthetic_dict"]["injectionLocation"])
    mu_grid = None

    #emissionNames = ['counterClock_rev0']
    for emissionName in emissionNames:
        rad_distribution[emissionName] = {}

        # Find the phi limits that the radDist was created with
        rD_phi, rD_P_pol = _bestFit_radDist_parameters(emissionName)
        rD_scale = _bestFit_radDist_scaling(emissionName)

        # Create total power arrays
        if 'phi' not in rad_distribution:
            rad_distribution['phi'] = rD_phi
            rad_distribution['total_power'] = np.zeros(rD_phi.shape)
        else:
            # Checker to make sure that phi arrays match when doing helical distributions
            # (or more than one injection location)
            if not np.array_equal(rD_phi, rad_distribution['phi']):
                raise ValueError("Error! PHI arrays do not match in _post_process_calculations")
                
        # dphi already accounts for _rev1, _rev2, etc. for helical distributions
        # Snap mu to the phi grid, so mu is zero at the injection location
        ang = (rD_phi - mu + np.pi) % (2.0 * np.pi) - np.pi
        mu_grid = float(rD_phi[int(np.argmin(np.abs(ang)))])
        dphi = Util_emis3D.find_dphi(rD_phi, mu_grid, emissionName=emissionName)


        # Both directions share amplitude 'a'; their individual decay constant
        # 'b' controls how fast each falls off away from the injection location.
        a = params[f"a_{inj_loc_tag}"]
        b = params[f"b_{emissionName}_{inj_loc_tag}"]
        b = 1

        print(f"{emissionName} a = {a:.2f}, b = {b:.2e}")

        scale_ = Util_emis3D.scale_wrapper(
            a=a,
            b=b,
            phi=rD_phi,
            mu=mu_grid,
            scale_def=scale_def,
            emissionName=emissionName,
            dphi=dphi,
        )

        # --- Now calculate the radiation around the vessel due to the radDist
        rD_power =  rD_P_pol  * scale_ * rD_scale
        
        rad_distribution[emissionName]["phi"] = np.asarray(rD_phi)
        rad_distribution[emissionName]["multiplication_factor"] = np.asarray(scale_)
        rad_distribution[emissionName]["total_power"] = np.asarray(rD_power)
        rad_distribution['total_power'] += rD_power

    # --- Remove the spot at the injection location, since it is double-counted
    if mu_grid is not None:
        loc = np.abs(rad_distribution["phi"] - mu_grid).argmin()
        rad_distribution['total_power'] = np.delete(rad_distribution['total_power'], loc)
        rad_distribution['phi'] = np.delete(rad_distribution['phi'], loc)

    tp = rad_distribution['total_power']
    tpf = np.max(tp) / (simpson(tp, x = rad_distribution['phi']) / (2.0 * np.pi))
    rad_distribution['toroidal_peaking_factor'] = tpf



    # CHECKER, plot the scale factors
    if True:
        plt.ion()
        f = plt.figure()
        ax = f.add_subplot(111)
        for emissionName in emissionNames:
            ax.scatter(rad_distribution[emissionName]["phi"] , 
                    rad_distribution[emissionName]["total_power"] ,
                    label = emissionName)
        ax.plot(rad_distribution['phi'], rad_distribution['total_power'], linewidth = 2.0, color = 'black', label = 'Total Emission')
        ax.legend()
        ax.set_xlabel('phi [rad]')
        ax.set_ylabel('multiplication factor')
        plt.show()

