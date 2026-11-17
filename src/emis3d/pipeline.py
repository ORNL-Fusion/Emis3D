# pipeline.py
"""
The three everyday jobs, as functions (used by the `emis3d` command and by
scripts/make_radDists.py and scripts/run_emis3D.py):

    make_raddists(config)   build the radDists of a library config (emis3d.raddist_library)
    run(run_config)         fit every evaluation time, save figures and results
    check(run_config)       find problems in a run config and its files before a long run
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Building radDist libraries
# ---------------------------------------------------------------------------


def _library_path(config, tokamak: str | None, ws) -> Path:
    given = Path(os.path.expanduser(str(config)))
    if given.is_file():
        return given
    if tokamak is None:
        raise FileNotFoundError(f"radDist library config {config} not found; give a path, or --tokamak to look in the raddists location")
    path = ws.path("raddists", tokamak=tokamak) / str(config)
    if not path.is_file():
        raise FileNotFoundError(f"radDist library config not found: {path}")
    return path


def start_positions(lib, t, ws):
    """The (R, z) start positions for one type: its grid, kept inside the wall."""
    import emis3d.util_radDist as Util_radDist
    from emis3d.tokamak import Tokamak

    tok = Tokamak(tokamakName=lib.tokamakName, mode="Analysis", reflections=False, eqFileName=t.settings.get("eqFileName"), workspace=ws)
    grid = t.settings["grid"]
    if not grid.get("rLimits") or not grid.get("zLimits"):
        rz = Util_radDist.callRZGridTokamak(tok, num_r=grid["NumR"], num_z=grid["NumZ"])
    else:
        if tok.wall is None:
            raise RuntimeError(f"{lib.tokamakName} has no wall to place the start positions in")
        rz = Util_radDist.createRZGrid(r_limits=grid["rLimits"], z_limits=grid["zLimits"], num_r=grid["NumR"],
                                       num_z=grid["NumZ"], wallcurve=tok.wall["wallcurve"])
    if rz is None or len(rz) == 0:
        raise ValueError(f"types.{t.name}: no start positions inside the wall for its grid")
    return rz


# Tidy names for reports (the radDist classes use a few different ones)
_REPORT_NAMES = {"numTransists": "numTransits"}


def describe_shape(shape: dict) -> str:
    """One radDist's shape parameters, with the library config's names."""
    if "sourceFile" in shape:
        return f"{Path(str(shape['sourceFile'])).name}, timestep {shape['timestep']}"
    return ", ".join(f"{_REPORT_NAMES.get(k, k)} {v:g}" if isinstance(v, (int, float)) else f"{_REPORT_NAMES.get(k, k)} {v}"
                     for k, v in shape.items())


def _values(v) -> str:
    vals = v if isinstance(v, (list, tuple)) else [v]
    text = ", ".join(f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in vals[:8])
    return text + (f", ... ({len(vals)} values)" if len(vals) > 8 else "")


def _plan_lines(lib, plans, root, workers_override) -> list[str]:
    """What will be built, before anything is built."""
    lines = [f"radDist library: {lib.path} ({lib.tokamakName}, shot {lib.shot}), into {root / lib.folder}"]
    for t, jobs, info, shapes in plans:
        rays = {**{"pixelSamples": 1000, "integrationStep": 0.01, "method": "compiled"}, **(t.settings.get("rays") or {})}
        lines.append(f"  {t.name} ({t.distType}) -> {lib.folder_for(t)}")
        if t.is_simulation:
            lines.append(f"      files: {_values([Path(str(f)).name for f in (t.shape['files'] if isinstance(t.shape['files'], list) else [t.shape['files']])])}"
                         f" | timesteps: {_values(t.shape['timesteps'])}")
            lines.append(f"      {len(jobs)} radDists (one per file and timestep)")
        else:
            lines.append("      " + " | ".join(f"{k}: {_values(v)}" for k, v in t.shape.items()))
            combos = 1
            for v in t.shape.values():
                combos *= len(v) if isinstance(v, (list, tuple)) else 1
            dedup = f" (of {combos} combinations)" if combos != len(shapes) else ""
            if not t.uses_start_positions:
                lines.append(f"      {len(shapes)} distinct shape(s){dedup}, centred on the magnetic axis (no start positions) = {len(jobs)} radDists")
            else:
                g = t.settings["grid"]
                where = "inside the wall" if not g.get("rLimits") else f"R {g['rLimits']}, z {g['zLimits']}"
                lines.append(f"      {len(shapes)} distinct shape(s){dedup} x {info['start_positions']} start position(s) "
                             f"(grid {g['NumR']} x {g['NumZ']}, {where}) = {len(jobs)} radDists")
        if rays["method"] == "matrix":
            lines.append(f"      geometry matrix {rays.get('matrix', 'medium')!r} (no ray tracing); "
                         f"workers: {workers_override or t.settings.get('workers') or 1}")
        else:
            lines.append(f"      rays: {rays['pixelSamples']} per channel, step {rays['integrationStep']} m, {rays['method']} emission; "
                         f"workers: {workers_override or t.settings.get('workers') or 1}")
    lines.append(f"Total: {sum(len(p[1]) for p in plans)} radDists")
    return lines


def _build_one(task):
    """
    Worker: build and save one radDist, returning (seconds, error or None).
    A failure is reported, not raised, so the other radDists still build;
    make_raddists lists the failures at the end. Defined at module level so
    worker processes can import it.
    """
    import contextlib
    import io
    import time

    import emis3d.util_radDist as Util_radDist

    builder_name, job = task
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # the progress report is printed by the main process
            getattr(Util_radDist, builder_name)(job)
        return time.perf_counter() - t0, None
    except Exception as e:  # noqa: BLE001  boundary: any failure of one radDist is reported, the rest continue
        return time.perf_counter() - t0, f"{type(e).__name__}: {e}"


def _init_worker(ws) -> None:
    """
    Start of a radDist-building worker process: give it the workspace of the
    run that started it. The workers are started fresh ("spawn"), so nothing
    set in the parent process reaches them unless it is passed here.
    """
    from emis3d.workspace import set_workspace

    set_workspace(ws)


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def make_raddists(config, tokamak: str | None = None, workspace=None, only=None, workers: int | None = None,
                  dry_run: bool = False, progress: bool = False) -> dict:
    """
    Build the radDists of a library config (emis3d.raddist_library), every
    enabled type or only those named in `only`.

    config   : a path, or a name in the workspace's raddists/<tokamak> folder
    workers  : radDists built in parallel (default: the config's workers, else 1)
    dry_run  : only plan; nothing is built
    progress : print the plan first, a line per radDist as it finishes
               (count, parameters, start position, time, time left) and a
               summary at the end

    Each type is built into raddists/<tokamak>/<folder>/<type name>. A radDist
    that fails is reported and the others still build. After a build,
    manifest.yaml in raddists/<tokamak>/<folder> records the config, the
    counts, any failures and the code version. Returns
    {"types": [{name, distType, shapes, start_positions, radDists, folder}],
     "radDists": total, "built": n, "failed": [{type, parameters, error}], "seconds": t}.
    """
    import time
    from concurrent.futures import as_completed

    import yaml

    from emis3d.raddist_library import RadDistLibrary
    from emis3d.workspace import Workspace, set_workspace

    def say(text=""):
        if progress:
            print(text, flush=True)

    ws = workspace if isinstance(workspace, Workspace) else Workspace.resolve(workspace=workspace)
    set_workspace(ws)
    ws.export_env()  # worker processes resolve the same workspace

    path = _library_path(config, tokamak, ws)
    lib = RadDistLibrary.load(path)
    if tokamak and tokamak != lib.tokamakName:
        raise ValueError(f"{path} is for {lib.tokamakName}, not {tokamak}")
    selected = lib.select(only)
    root = ws.path("raddists", tokamak=lib.tokamakName)

    builders = {
        "ElongatedRing": "radDist_ElongatedRing_parallel",
        "Helical": "radDist_Helical_parallel",
        "HelicalRing": "radDist_HelicalRing_parallel",
        "SquareTube": "radDist_SquareTube_parallel",
        "FluxShell": "radDist_FluxShell_parallel",
        "NIMROD": "radDist_NIMROD_parallel",
        "M3DC1": "radDist_M3DC1_parallel",
    }

    # --- Types built through a geometry matrix need it in the cache
    from emis3d.geometry import matrix_path

    matrices = {}
    for t in selected:
        rays = t.settings.get("rays") or {}
        if rays.get("method") == "matrix":
            name = str(rays.get("matrix", "medium"))
            mpath = matrix_path(ws, lib.tokamakName, name)
            if not mpath.is_file():
                raise FileNotFoundError(f"types.{t.name}: no geometry matrix {name!r} for {lib.tokamakName} at {mpath}; "
                                        f"build it with: emis3d build-matrix {lib.tokamakName} --resolution {name}")
            matrices[t.name] = mpath

    plans = []
    for t in selected:
        shapes = lib.shape_sets(t)

        def cfg_for(shape, t=t):
            cfg = lib.radDist_config(t, shape)
            if t.name in matrices:
                cfg["matrixPath"] = str(matrices[t.name])
            return cfg

        if t.is_simulation:
            jobs = [((shape["sourceFile"], shape["timestep"], cfg_for(shape)), shape, None) for shape in shapes]
            positions = 1
        elif not t.uses_start_positions:
            # One radDist per shape: it is the same whatever the start position (the builder
            # still takes one; the radDist replaces it with the magnetic axis). Looping over
            # the grid built the identical radDist once per position, into the same file.
            jobs = [(((0.0, 0.0), cfg_for(shape)), shape, None) for shape in shapes]
            positions = 1
        else:
            rz = start_positions(lib, t, ws)
            jobs = [((val, cfg_for(shape)), shape, val) for shape in shapes for val in rz]
            positions = len(rz)
        info = {"name": t.name, "distType": t.distType, "shapes": len(shapes), "start_positions": positions,
                "radDists": len(jobs), "folder": str(root / lib.folder_for(t))}
        plans.append((t, jobs, info, shapes))

    for line in _plan_lines(lib, plans, root, workers):
        say(line)
    summary = {"types": [p[2] for p in plans], "radDists": sum(p[2]["radDists"] for p in plans), "built": 0, "failed": [], "seconds": 0.0}
    if dry_run:
        say("Dry run: nothing built")
        return summary

    total, done, start = summary["radDists"], 0, time.perf_counter()
    width = len(str(total))

    def report(t, jobs, k, seconds, error):
        """One line per finished radDist; records failures."""
        nonlocal done
        done += 1
        _, shape, rz = jobs[k]
        where = f" | R {rz[0]:.3f}, z {rz[1]:.3f}" if rz is not None else ""
        elapsed = time.perf_counter() - start
        left = elapsed / done * (total - done)
        status = f"FAILED: {error}" if error else f"{seconds:.1f} s"
        say(f"[{done:>{width}}/{total} {100 * done / total:5.1f}%] {t.name}: {describe_shape(shape)}{where} | {status}"
            f" | elapsed {_clock(elapsed)}, left ~{_clock(left)}")
        if error:
            summary["failed"].append({"type": t.name, "parameters": describe_shape(shape) + where, "error": error})
        else:
            summary["built"] += 1

    for t, jobs, _, _ in plans:
        n_workers = max(1, int(workers or t.settings.get("workers") or 1))
        say(f"\nBuilding {t.name}: {len(jobs)} radDists with {n_workers} worker(s)")
        tasks = [(builders[t.distType], job) for job, _, _ in jobs]
        if n_workers == 1:
            for k, task in enumerate(tasks):
                report(t, jobs, k, *_build_one(task))
        else:
            # Workers are started fresh ("spawn") on every platform, as macOS and Windows do
            # and as Util_emis3D.FitPool does: forking a process that already runs threads
            # (numba's, an interpreter's in a notebook) can deadlock the child, and Python
            # 3.14 stops forking by default. Each worker is handed the workspace itself, so
            # it uses the same locations however the workspace was given (an argument, a
            # run config's paths, overrides), not only what the environment says.
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_init_worker, initargs=(ws,)) as executor:
                futures = {executor.submit(_build_one, task): k for k, task in enumerate(tasks)}
                for fut in as_completed(futures):
                    report(t, jobs, futures[fut], *fut.result())

    summary["seconds"] = time.perf_counter() - start
    manifest = {
        "config": str(path.resolve()),
        "matrices": {name: str(p) for name, p in matrices.items()},
        "built": summary["types"],
        "failed": summary["failed"],
        "seconds": round(summary["seconds"], 1),
        "provenance": ws.provenance(),
        "library": yaml.safe_load(path.read_text()),
    }
    out = root / lib.folder / "manifest.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(manifest, sort_keys=False))

    say(f"\nBuilt {summary['built']} of {total} radDists in {_clock(summary['seconds'])}"
        + (f"; {len(summary['failed'])} failed:" if summary["failed"] else ""))
    for f in summary["failed"]:
        say(f"  {f['type']}: {f['parameters']}: {f['error']}")
    say(f"Manifest: {out}")
    return summary


# ---------------------------------------------------------------------------
# Geometry matrices
# ---------------------------------------------------------------------------


def build_matrix(machine: str, cameras=None, resolution: str = "medium", samples: int | None = None, workspace=None,
                 name: str | None = None, progress: bool = False) -> Path:
    """
    Build the geometry matrix of a machine's cameras (emis3d.geometry) and
    save it in the workspace's cache as <cache>/<machine>/geometry/<name>.npz
    (name defaults to the resolution, plus the camera names when a subset).
    Returns the file's path.
    """
    from emis3d.geometry import build_geometry_matrix, matrix_path
    from emis3d.tokamak import Tokamak
    from emis3d.workspace import Workspace, set_workspace

    ws = workspace if isinstance(workspace, Workspace) else Workspace.resolve(workspace=workspace)
    set_workspace(ws)
    say = (lambda text: print(text, flush=True)) if progress else (lambda text: None)

    tok = Tokamak(tokamakName=machine, mode="Build", reflections=False, loadBolometers=True, workspace=ws)
    if tok.info is None:
        raise RuntimeError(f"machine {machine} did not load")
    G = build_geometry_matrix(tok, cameras=cameras, resolution=resolution, samples=samples, progress=say)
    G.provenance.update(ws.provenance())
    if name is None:
        name = resolution + ("_" + "_".join(G.cameras) if cameras else "")
    out = G.save(matrix_path(ws, machine, name))
    say(f"saved {out} ({len(G.channels)} channels, {G.rows.nnz:,} nonzeros, built in {G.provenance.get('built_in', 0):.0f} s)")
    return out


def build_mapping(machine: str, matrix: str, equilibrium: str, injection_deg: float, transits: int = 1, substeps: int = 2,
                  psi_max: float | None = None, workspace=None, progress: bool = False) -> Path:
    """
    Build (or find) the field-line mapping of a matrix for an equilibrium and
    injector angle (emis3d.fieldline_map), cached beside the matrix. The
    helical parametric model builds it itself when missing; this command
    does it ahead of time. Returns the file's path.
    """
    from emis3d.fieldline_map import get_or_build
    from emis3d.geometry import cached_matrix, matrix_path
    from emis3d.workspace import Workspace, set_workspace

    ws = workspace if isinstance(workspace, Workspace) else Workspace.resolve(workspace=workspace)
    set_workspace(ws)
    say = (lambda text: print(text, flush=True)) if progress else (lambda text: None)
    mpath = matrix_path(ws, machine, matrix)
    if not mpath.is_file():
        raise FileNotFoundError(f"no geometry matrix {matrix!r} at {mpath}; build it with: emis3d build-matrix {machine} --resolution {matrix}")
    fmap, path = get_or_build(ws, cached_matrix(mpath), machine, matrix, equilibrium, injection_deg, transits, substeps, psi_max, progress=say)
    say(f"mapping: {path} ({fmap.n_pixels} source pixels, {len(fmap.passes)} passes)")
    return path


# ---------------------------------------------------------------------------
# Running an analysis
# ---------------------------------------------------------------------------


_TIME_WORKER: dict = {}  # a time-fitting worker process's own Emis3D object, and the time it is fitting


class _TimePrefix(logging.Filter):
    """Starts a worker's log messages with the time it is fitting: several workers write to the same terminal."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _TIME_WORKER.get("time") is not None:
            record.msg = f"[t = {_TIME_WORKER['time']:g}] {record.msg}"
        return True


def _init_time_worker(run_config: str, ws) -> None:
    """
    Start of a time-fitting worker process (run, with solver: parametric):
    its own Emis3D object for the run config, made once and used for every
    time the worker is given. What the object prints (the banner, the fit
    reports) is dropped, since the workers would write through each other;
    warnings are shown, with the time they belong to.
    """
    from emis3d.emis3D import Emis3D
    from emis3d.workspace import set_workspace

    set_workspace(ws)
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115 (open for the life of the process)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    handler.addFilter(_TimePrefix())
    logging.getLogger().addHandler(handler)
    e = Emis3D(runConfigName=run_config, workspace=ws)
    e.info["numProcessorsFitting"] = 1  # the fitted shape's own amplitude fit stays in this process
    _TIME_WORKER["emis3d"] = e


def _fit_time(t: float) -> tuple[float, str, float]:
    """Fit one time in a worker process and save it as a checkpoint: (time, checkpoint file, seconds)."""
    e = _TIME_WORKER["emis3d"]
    _TIME_WORKER["time"] = t
    start = time.perf_counter()
    e._perform_fits(evalTime=t, cleanup=True)
    path = e._save_checkpoint(t)
    if path is None:
        raise RuntimeError("the run config needs a shot and a tokamakName to save a time")
    for store in ("bestFits", "fitData", "fits", "parametric", "parametric_models", "_parametric_fits", "_parametric_radDist", "_scale_override"):
        getattr(e, store, {}).pop(t, None)  # the time is in the file: free its memory for the next one
    return t, str(path), time.perf_counter() - start


def _fit_times_in_parallel(e, run_config: str, times: list, workers: int, keep_checkpoints: bool) -> None:
    """
    Fit the times of a parametric run in worker processes, one time each at a
    time, and gather them into e.bestFits and e.fitData.

    A time's fit depends on that time's data alone (the search starts from
    the run config's seed at every time), so the results are those of the
    serial loop. Each worker saves its time as a checkpoint
    (Emis3D._save_checkpoint), which is how the result comes back: an
    interrupted run keeps the finished times, as a serial one does. If a time
    fails, the times not yet started are dropped, the ones being fitted are
    finished and kept, and the error is raised.
    """
    e._build_parametric_mappings()  # once, here: not by every worker at the same moment
    n = min(int(workers), len(times))
    width = len(str(len(times)))
    print(f"Fitting {len(times)} times in {n} worker processes", flush=True)
    start, fitting = time.perf_counter(), 0.0
    files: dict = {}
    # Started fresh ("spawn"), as the radDist-building workers and Util_emis3D.FitPool are
    with ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_time_worker, initargs=(run_config, e.ws)) as executor:
        futures = {executor.submit(_fit_time, float(t)): float(t) for t in times}
        try:
            for done, fut in enumerate(as_completed(futures), start=1):
                t, path, seconds = fut.result()
                files[t] = path
                fitting += seconds
                left = fitting / done * (len(times) - done) / n  # the mean time of a fit, for what is left, shared by the workers
                print(f"[{done:>{width}}/{len(times)}] t = {t:g}: fitted in {_clock(seconds)}"
                      f" | elapsed {_clock(time.perf_counter() - start)}, left ~{_clock(left)}", flush=True)
        except BaseException:
            running = sum(f.running() for f in futures)
            print(f"A time failed: finishing the {running} being fitted, dropping the rest", file=sys.stderr, flush=True)
            executor.shutdown(wait=True, cancel_futures=True)
            raise
    for t in times:  # in the order of the times, as the serial loop leaves them
        e._add_results(files[float(t)])
        if not keep_checkpoints:
            Path(files[float(t)]).unlink(missing_ok=True)


def run(run_config, workspace=None, workers: int | None = None, times=None, plots: bool = True, save: bool = True):
    """
    Fit every evaluation time (the run config's tstart/tend/Nt, or `times`),
    save the best fits and one figure per time. Returns the Emis3D object.

    workers (default: the run config's numProcessorsFitting) is the number
    of processes fitting at once. With solver: library they share the
    radDists of a time (Util_emis3D.FitPool); with solver: parametric each
    fits a whole time, so several times are fitted at once
    (_fit_times_in_parallel). One time, or one worker, is fitted in this
    process.

    The best fits are saved before the figures are drawn, so a figure that
    fails does not lose the fits. With save, every time is also saved on
    its own as soon as it is finished (Emis3D._save_checkpoint), so a run
    that stops part of the way through keeps the times it finished, in
    results/<run_id>/checkpoints; Emis3D._load_bestFits reads that folder.
    """
    from emis3d.emis3D import Emis3D

    e = Emis3D(runConfigName=str(run_config), workspace=workspace)
    if workers is not None:
        e.info["numProcessorsFitting"] = int(workers)
    after_each = e._save_checkpoint if save else None
    n_workers = int((e.info or {}).get("numProcessorsFitting", 1) or 1)
    todo = []  # the times to share between worker processes, if this run does that
    if n_workers > 1 and e.config is not None and e.config.solver == "parametric":
        todo = e.eval_times() if times is None else [float(t) for t in times]
    try:
        if len(todo) > 1:
            _fit_times_in_parallel(e, str(run_config), todo, n_workers, keep_checkpoints=save)
        elif times is None:
            e._perform_fits(after_each=after_each)
        else:
            for t in times:
                e._perform_fits(evalTime=float(t), cleanup=True, after_each=after_each)
    finally:
        e.close()
    if save:
        e._save_bestFits()
    if plots:
        e._plot_bestFit(save=True)
    return e


# ---------------------------------------------------------------------------
# Checking a run config before a long run
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    level: str  # "ok", "warning" or "error"
    message: str


def check(run_config, workspace=None, sample: int = 3) -> list[Finding]:
    """
    Problems in a run config and the files it refers to, without fitting:
    the config itself, the machine and cameras, every data file (readable,
    channels, bad channels, evaluation times inside the data), and every
    radDist folder (present, files of the current format, channels that match
    the data, injection location). Loads only `sample` radDists per folder.
    """
    from emis3d.globals import RADDIST_FORMAT
    from emis3d.run_config import RunConfig
    from emis3d.util import config_loader, load_json, read_h5
    from emis3d.workspace import Workspace, WorkspaceError

    out: list[Finding] = []

    def ok(msg):
        out.append(Finding("ok", msg))

    def warn(msg):
        out.append(Finding("warning", msg))

    def err(msg):
        out.append(Finding("error", msg))

    path = Path(os.path.expanduser(str(run_config))).resolve()
    if not path.is_file():
        err(f"run config not found: {path}")
        return out
    try:
        raw = config_loader(path)
        cfg = RunConfig.from_dict(raw if isinstance(raw, dict) else {})
    except (TypeError, ValueError) as e:
        err(f"run config: {e}")
        return out
    ok(f"run config is valid: {path.name}")

    if not cfg.tokamakName:
        err("tokamakName is not set")
        return out
    try:
        ws = workspace if isinstance(workspace, Workspace) else Workspace.resolve(workspace=workspace, run_config=path, run_paths=cfg.paths)
    except (WorkspaceError, ValueError, OSError) as e:
        err(f"workspace: {e}")
        return out
    ok(f"workspace: {ws.fields.get('workspace', ws.layout)}")

    times = cfg.eval_times()
    if times:
        ok(f"{len(times)} evaluation time(s): {times[0]:g} to {times[-1]:g}")
    else:
        warn("no evaluation times: set tstart (and tend, Nt), or pass times when running")
    if cfg.time_grid_problem():
        warn(f"evaluation times: {cfg.time_grid_problem()}")

    # --- Machine and cameras
    if not ws.has_machine(cfg.tokamakName):
        err(f"machine {cfg.tokamakName} not found (emis3d machines lists what is available)")
        return out
    from emis3d.tokamak import Tokamak

    try:
        tok = Tokamak(tokamakName=cfg.tokamakName, mode="Analysis", loadBolometers=True, workspace=ws)
    except (FileNotFoundError, ValueError) as e:  # a camera config that is missing or incomplete
        err(f"machine {cfg.tokamakName}: {e}")
        return out
    groups = {b.group_name for b in tok.bolometers}
    unknown = sorted(set(cfg.BOLOMETERS) - groups)
    if unknown:
        err(f"cameras not in the {cfg.tokamakName} machine files: {unknown} (machine has {sorted(groups)})")
    else:
        ok(f"cameras found: {sorted(cfg.BOLOMETERS)}")

    # --- Data files
    data_channels = set()
    meas = ws.path("measurements", tokamak=cfg.tokamakName, shot=cfg.shot)
    for cam, entry in cfg.BOLOMETERS.items():
        f = meas / str((entry or {}).get("dataFileName", ""))
        if not f.is_file():
            err(f"{cam}: data file not found: {f}")
            continue
        try:
            h = read_h5(f)  # the same reader a run uses
        except OSError as e:
            err(f"{cam}: cannot read {f.name}: {e}")
            continue
        missing = [k for k in ("TIME", "DATA_CALIBRATED", "channelOrder") if k not in h]
        if missing:
            err(f"{cam}: {f.name} lacks {missing}")
            continue
        t = np.asarray(h["TIME"], dtype=float)
        chans = [c.decode() if isinstance(c, bytes) else str(c) for c in np.atleast_1d(h["channelOrder"])]
        data_channels |= set(chans)
        bad = set((entry or {}).get("badChannels", [])) - set(chans)
        if bad:
            err(f"{cam}: badChannels not in the data file: {sorted(bad)}")
        dt = cfg.dt or 0.0
        outside = [x for x in times if x - dt < t.min() or x + dt > t.max()]
        if outside:
            err(f"{cam}: time(s) {outside} (dt {dt:g}) outside the data, which runs from {t.min():g} to {t.max():g}")
        else:
            ok(f"{cam}: {f.name}, {len(chans)} channels, data from {t.min():g} to {t.max():g}")

    ok(f"solver: {cfg.solver}; bounds a {cfg.amplitude_bounds()}, b {cfg.decay_bounds()}")

    # --- Can the cameras determine the toroidal profile?
    from emis3d.diagnostic import camera_angles

    inj_deg = cfg.injectionLocation
    if cfg.injector:
        try:
            from emis3d.injector import machine_injectors

            found = machine_injectors(ws, cfg.tokamakName).get(cfg.injector)
            inj_deg = found.phi_deg if found is not None else inj_deg
        except (AttributeError, ValueError):
            pass
    problem = cfg.toroidal_identifiability(camera_angles(ws, cfg.tokamakName), inj_deg)
    if problem:
        warn(problem)
    else:
        ok(f"toroidal profile: {len(cfg.toroidal_unknowns())} free parameter(s) ({', '.join(cfg.toroidal_unknowns()) or 'none'})")

    # --- The SPI injector, if the run names one
    if cfg.injector:
        from emis3d.injector import machine_injectors

        try:
            injectors = machine_injectors(ws, cfg.tokamakName)
        except ValueError as e:
            err(str(e))
            injectors = {}
        inj = injectors.get(cfg.injector)
        if inj is None:
            err(f"injector {cfg.injector!r} is not in {cfg.tokamakName}'s settings (INJECTORS: {sorted(injectors) or 'none'})")
        else:
            given = raw.get("injectionLocation") if isinstance(raw, dict) else None
            if given is not None and abs(((float(given) - inj.phi_deg + 180.0) % 360.0) - 180.0) > 0.5:
                err(f"injectionLocation {given} does not match injector {inj.name}'s phi_deg {inj.phi_deg:g}")
            (ok if inj.verified else warn)(inj.describe())

    # --- Parametric fit: its geometry matrices must be in the cache
    if cfg.solver == "parametric":
        from emis3d.geometry import matrix_path

        for name in cfg.parametric.get("matrices") or ["medium"]:
            mpath = matrix_path(ws, cfg.tokamakName, name)
            if mpath.is_file():
                ok(f"geometry matrix {name!r}: {mpath}")
            else:
                err(f"geometry matrix {name!r} not found at {mpath}; build it with: emis3d build-matrix {cfg.tokamakName} --resolution {name}")
        ok(f"parametric models: {', '.join(cfg.parametric['models'])}")

    # --- radDist folders (the library solver's)
    base = ws.path("raddists", tokamak=cfg.tokamakName)
    folders = [(d, "LocIndependent") for d in cfg.radDistDirectories_LocIndependent] if cfg.solver == "library" else []
    folders += [(d, "LocDependent") for d in cfg.radDistDirectories_LocDependent] if cfg.solver == "library" else []
    for folder, kind in folders:
        p = base / folder
        files = sorted(q for q in p.rglob("*.json")) if p.is_dir() else []
        if not p.is_dir():
            err(f"radDist folder not found: {p}")
            continue
        if not files:
            err(f"radDist folder is empty: {p}")
            continue
        problems = []
        for q in files[:sample]:
            try:
                saved = load_json(str(q))
                info, data = saved["info"], saved["data"]
            except (OSError, ValueError, KeyError) as e:
                problems.append(f"{q.name}: unreadable ({e})")
                continue
            if info.get("radDistFormat") != RADDIST_FORMAT:
                problems.append(f"{q.name}: format {info.get('radDistFormat')}, not {RADDIST_FORMAT}: regenerate")
            units = info.get("units")
            rd_channels = {c for chans in data.get(units, {}).get("channelOrder", {}).values() for c in chans}
            if data_channels and not (data_channels & rd_channels):
                problems.append(f"{q.name}: no channels in common with the data")
            # The run's injection location: the named injector's angle when the config gives
            # injector:, else injectionLocation (as Emis3D resolves it), compared as angles
            rd_inj = info.get("injectionLocation")
            if kind == "LocDependent" and rd_inj is not None and abs(((float(rd_inj) - float(inj_deg) + 180.0) % 360.0) - 180.0) > 1e-6:
                whose = f"injector {cfg.injector}'s {float(inj_deg):g}" if cfg.injector else f"the run config's {cfg.injectionLocation}"
                problems.append(f"{q.name}: injectionLocation {rd_inj} differs from {whose}")
        if problems:
            for msg in problems:
                err(f"{folder}: {msg}")
        else:
            ok(f"{folder}: {len(files)} radDist file(s); first {min(sample, len(files))} checked")
    return out
