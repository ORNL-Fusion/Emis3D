# run_emis3D.py
"""
Code that runs emis3D in parallel

Written by JLH Sept. 2025
"""


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from concurrent.futures import ProcessPoolExecutor

import time
import numpy as np
import main.Emis3D as Emis3D
import main.Util_emis3D as Util_emis3D


if __name__ == "__main__":

    # --- Update these parameters:
    evalTimes = [2124.0]  # np.arange(2119, 2127, 0.3)
    tokamakName = "DIII-D"
    runConfigName = "184407/184407_runConfig.yaml"
    verbose = True

    # ----- No need to update anything below
    t = Emis3D.Emis3D(
        tokamakName=tokamakName, runConfigName=runConfigName, verbose=verbose
    )

    for evalTime in evalTimes:
        t._prepare_fits(evalTime=evalTime, crossCalib=False)

        jobs = []
        data_dict = t.fitData[evalTime]
        scale_def = "von_mises"
        max_workers = 1
        if t.info is not None:
            if "scale_def" in t.info:
                scale_def = t.info["scale_def"]
            if "numProcessorsFitting" in t.info:
                max_workers = t.info["numProcessorsFitting"]

        for ii in t.fits[evalTime]:
            if isinstance(ii, int):
                jobs.append(
                    (
                        ii,
                        t.fits[evalTime][ii]["parameters"],
                        data_dict,
                        t.fits[evalTime][ii]["synthetic_dict"],
                        scale_def,
                    )
                )
        results = {}
        start_time = time.time()
        print("Preforming fits")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for ii, fit_result in executor.map(Util_emis3D.runParallel, jobs):
                results[ii] = fit_result
                if ii % 1_000 == 0 and t.verbose:
                    if t.info is not None and "numFits" in t.info:
                        print(f"Done with fit {ii} out of {t.info['numFits']}")

        print(f"Done with fits in {time.time() - start_time:.2f} seconds")

        # Merge results back into boss data
        for ii, fit_result in results.items():
            if fit_result is not None:
                t.fits[evalTime][ii]["fit"] = fit_result
                t.fits[evalTime]["chiSqVec"][ii] = float(fit_result.chisqr.item())

        # --- Preform post-processing
        t._post_process_fit_arrangement(evalTime=evalTime)
        t._post_process_radiation_distribution(evalTime=evalTime)
        t._post_process_calculations(evalTime=evalTime)
        t._cleanup_fits(evalTime=evalTime)  # to save memory
        t._plot_bestFit(evalTime=evalTime, save=True)

    # --- Save the best fits and the fit data after everything is done
    t._save_bestFits()
