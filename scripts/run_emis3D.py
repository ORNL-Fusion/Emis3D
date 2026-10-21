# run_emis3D.py
"""
Code that runs emis3D in parallel

Example call:
python run_emis3D.py

Written by JLH Sept. 2025
"""



# from concurrent.futures import ProcessPoolExecutor # ProcessPoolExecutor raises a lot of errors on a mac
from concurrent.futures import ThreadPoolExecutor

import time
from emis3d.emis3D import Emis3D
import emis3d.util_emis3D as Util_emis3D

_global_data_dict = None


def init_worker(data):
    """
    Initilizer for each worker process, sets the global data dict
    """
    global _global_data_dict
    _global_data_dict = data


def runParallel_with_global(job):
    """
    Wrapper that uses the global data_dict
    """
    fit_index, pars, synth_dict, scale_def, helical_endpoint_weight, fit_method = job

    return Util_emis3D.runParallel(
        (
            fit_index,
            pars,
            _global_data_dict,
            synth_dict,
            scale_def,
            helical_endpoint_weight,
            fit_method,
        )
    )


if __name__ == "__main__":

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

    for evalTime in evalTimes:
        t._prepare_fits(evalTime=evalTime)

        jobs = []
        data_dict = t.fitData[evalTime]
        # --- Defaults come from RunConfig
        scale_def = t.info["scale_def"]
        max_workers = t.info["numProcessorsFitting"]
        helical_endpoint_weight = t.info["helical_endpoint_weight"]
        fit_method = t.info["fit_method"]

        for ii in t.fits[evalTime]:
            if isinstance(ii, int):
                jobs.append(
                    (
                        ii,
                        t.fits[evalTime][ii]["parameters"],
                        t.fits[evalTime][ii]["synthetic_dict"],
                        scale_def,
                        helical_endpoint_weight,
                        fit_method,
                    )
                )
        results = {}
        start_time = time.time()
        start_time0 = time.time()
        print("→ Preforming fits")

        with ThreadPoolExecutor(
            max_workers=max_workers, initializer=init_worker, initargs=(data_dict,)
        ) as executor:
            for ii, fit_result in executor.map(runParallel_with_global, jobs):
                results[ii] = fit_result
                if ii % 1_000 == 0 and t.verbose:
                    if t.info is not None and "numFits" in t.info:
                        print(f"→ Done with fit {ii} out of {t.info['numFits']}")

        print(f"→ Done with fits in {time.time() - start_time:.2f} seconds")

        # Merge results back into boss data
        for ii, fit_result in results.items():
            if fit_result is not None:
                t.fits[evalTime][ii]["fit"] = fit_result
                t.fits[evalTime]["scores"][ii] = Util_emis3D.fit_score(fit_result)

        # --- Preform post-processing
        start_time = time.time()
        t._post_process_fit_arrangement(evalTime=evalTime)
        t._post_process_bolo_locations(evalTime=evalTime)
        t._post_process_calculations(evalTime=evalTime)
        print(f"→ Done with postprocessing in {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        t._cleanup_fits(evalTime=evalTime)  # to save memory
        t._plot_bestFit(evalTime=evalTime, save=True)
        print(
            f"→ Program completed for evalTime {evalTime:.4f} in {time.time() - start_time0:.2f} seconds\n"
        )

    # --- Save the best fits and the fit data after everything is done
    t._save_bestFits()
