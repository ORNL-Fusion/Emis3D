# run_emis3D.py
"""
Runs Emis3D for every evaluation time in the run config (tstart, tend, Nt):
fits in parallel (numProcessorsFitting), saves one figure per time and the
best fits.

Equivalent command: emis3d run <run config> [--times ...] [--workers N]

Written by JLH Sept. 2025
"""

import time

from emis3d.pipeline import run

if __name__ == "__main__":  # required: worker processes re-import this file on macOS
    # --- Update this: a path, or a name relative to the workspace's runs folder
    runConfigName = "~/projects/emis3d-runs/runs/JET/95709/95709_runConfig.yaml"

    start = time.time()
    e = run(runConfigName)
    for t in sorted(e.bestFits):
        print(f"t = {t:g}: TPF {e.bestFits[t]['radiation_distribution']['toroidal_peaking_factor']:.3f}")
    print(f"→ Done in {time.time() - start:.1f} s; results in {e._results_dir()}")
