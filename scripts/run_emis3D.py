# run_emis3D.py
"""
Runs Emis3D for every evaluation time in the run config (tstart, tend, Nt),
fitting in parallel, and saves one figure per time and the best fits.

The fits run in worker processes (numProcessorsFitting in the run config);
the pool is started once and reused for every evaluation time.

Example call:
    python run_emis3D.py

Written by JLH Sept. 2025
"""

import time

from emis3d.emis3D import Emis3D

if __name__ == "__main__":  # required: worker processes re-import this file on macOS
    # --- Update these parameters (the evaluation times come from tstart, tend
    # --- and Nt in the run config):
    tokamakName = "JET"
    runConfigName = "JET/95709/95709_runConfig.yaml"  # relative to the workspace's runs folder
    verbose = True

    # ----- No need to update anything below
    t = Emis3D(tokamakName=tokamakName, runConfigName=runConfigName, verbose=verbose)
    times = t.eval_times()
    print(f"→ {len(times)} evaluation time(s): {times}")
    print(f"→ Fitting with {t.info['numProcessorsFitting']} worker process(es)")

    start_time = time.time()
    try:
        t._perform_fits()  # every time, keeping only each time's best fit
    finally:
        t.close()  # stop the worker processes
    print(f"→ Fits and post-processing done in {time.time() - start_time:.2f} seconds")

    t._plot_bestFit(save=True)  # one PNG per time
    t._save_bestFits()
