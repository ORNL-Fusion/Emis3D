# helical_endpoint_testing.py
"""
Testing the helical endpoint continuity constraint

"""



import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from emis3d.Emis3D import Emis3D
from emis3d.Util_emis3D import _emission_pairs, loc_tag,scale_wrapper



evalTime = 50.953
tokamakName = "JET"
runConfigName = "95709/95709_runConfig.yaml"
verbose = True

t = Emis3D(
    tokamakName=tokamakName,
    runConfigName=runConfigName,
    verbose=verbose,
    initialize=True,
)
t._prepare_fits(evalTime=evalTime, crossCalib=False)



synthetic_dict = t.fits[evalTime][0]['synthetic_dict']


if True:
    weight = 1.0



    pairs = _emission_pairs(synthetic_dict.get("emissionNames", []))

    pars = t.fits[evalTime][0]['parameters']
    params = pars.valuesdict()

    tag = loc_tag(synthetic_dict["injectionLocation"])

    a_name = f"a_{tag}"

    a = params[a_name]


    # Injection / peak location (rad)
    mu = float(synthetic_dict["injectionLocation_rad"])
    if "peak_rad_loc" in params:
        mu = float(params["peak_rad_loc"])

    two_pi = 2.0 * np.pi


    eps = 1.0e-30
    penalties = []

    for cw_name, ccw_name in pairs:
        endpoint_power = {}
        peak_Ppol = {}
        usable = True

        for name in (cw_name, ccw_name):
            sub = synthetic_dict.get(name, {})
            phi_arr = np.asarray(sub.get("phi_array", []), dtype=float)
            P_pol = np.asarray(sub.get("P_pol", []), dtype=float)
            if phi_arr.size == 0 or P_pol.size == 0:
                usable = False
                break

            b = params.get(f"b_{name}_{tag}", 0.0)

            # Scale evaluated at the endpoint winding distance (NOT at phi = mu,
            # which would return the dphi = 0 peak).
            scale_end = scale_wrapper(
                a,
                b,
                phi=phi_arr,
                mu=mu,
                scale_def='linear',
                emissionName=name,
            )

            if scale_end is None:
                usable = False
                break

            # P_pol sampled at the injection toroidal location (mod 2*pi)
            idx = int(np.argmin(np.abs(phi_arr - mu)))
            scaleSynth = float(sub.get("scaleSynth", 1.0))

            endpoint_power[name] = scale_end[idx] * P_pol[idx] * scaleSynth
            peak_Ppol[name] = P_pol[idx] * scaleSynth

        if not usable:
            continue

        p_cw = endpoint_power[cw_name]
        p_ccw = endpoint_power[ccw_name]

        # Peak power shared reference (dphi = 0 amplitude is 'a'); keeps the
        # penalty bounded and ->0 when both endpoints have decayed away.
        p_peak = a * 0.5 * (peak_Ppol[cw_name] + peak_Ppol[ccw_name])
        penalties.append(weight * (p_cw - p_ccw) / (p_peak + eps))

