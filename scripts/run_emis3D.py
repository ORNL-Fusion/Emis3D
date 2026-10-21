# run_emis3D.py
"""
Runs Emis3D for several evaluation times, fitting in parallel.

The fits run in worker processes (numProcessorsFitting in the run config);
the pool is started once and reused for every evaluation time.

Example call:
    python run_emis3D.py

Written by JLH Sept. 2025
"""

import time

from emis3d.emis3D import Emis3D

if __name__ == "__main__":  # required: worker processes re-import this file on macOS
    # --- Update these parameters:
    evalTimes = [
        # 50.949,
        # 50.951,
        # 50.953,
        # 50.95,
        # 50.955,
        # 50.9556,
        50.9569,
        # 50.9627,
    ]
    tokamakName = "JET"
    runConfigName = "95709/95709_runConfig.yaml"
    verbose = True

    # ----- No need to update anything below
    t = Emis3D(tokamakName=tokamakName, runConfigName=runConfigName, verbose=verbose)
    print(f"→ Fitting with {t.info['numProcessorsFitting']} worker process(es)")

    try:
        for evalTime in evalTimes:
            start_time0 = time.time()
            t._prepare_fits(evalTime=evalTime)

            start_time = time.time()
            print("→ Performing fits")
            t._minimize_radDists(evalTime=evalTime)
            print(f"→ Done with fits in {time.time() - start_time:.2f} seconds")

            # --- Post-processing
            start_time = time.time()
            t._post_process_fit_arrangement(evalTime=evalTime)
            t._post_process_bolo_locations(evalTime=evalTime)
            t._post_process_calculations(evalTime=evalTime)
            t._post_process_uncertainty(evalTime=evalTime)  # before _cleanup_fits
            print(f"→ Done with postprocessing in {time.time() - start_time:.2f} seconds")

            t._cleanup_fits(evalTime=evalTime)  # to save memory
            t._plot_bestFit(evalTime=evalTime, save=True)
            print(f"→ Program completed for evalTime {evalTime:.4f} in {time.time() - start_time0:.2f} seconds\n")
    finally:
        t.close()  # stop the worker processes

    # --- Save the best fits and the fit data after everything is done
    t._save_bestFits()
