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
import os
from concurrent.futures import ProcessPoolExecutor
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
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
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


# ---------------------------------------------------------------------------
# Running an analysis
# ---------------------------------------------------------------------------


def run(run_config, workspace=None, workers: int | None = None, times=None, plots: bool = True, save: bool = True):
    """
    Fit every evaluation time (the run config's tstart/tend/Nt, or `times`),
    save one figure per time and the best fits. Returns the Emis3D object.
    """
    from emis3d.emis3D import Emis3D

    e = Emis3D(runConfigName=str(run_config), workspace=workspace)
    if workers is not None:
        e.info["numProcessorsFitting"] = int(workers)
    try:
        if times is None:
            e._perform_fits()
        else:
            for t in times:
                e._perform_fits(evalTime=float(t), cleanup=True)
    finally:
        e.close()
    if plots:
        e._plot_bestFit(save=True)
    if save:
        e._save_bestFits()
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

    # --- Machine and cameras
    if not ws.has_machine(cfg.tokamakName):
        err(f"machine {cfg.tokamakName} not found (emis3d machines lists what is available)")
        return out
    from emis3d.tokamak import Tokamak

    tok = Tokamak(tokamakName=cfg.tokamakName, mode="Analysis", loadBolometers=True, workspace=ws)
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

    # --- radDist folders
    base = ws.path("raddists", tokamak=cfg.tokamakName)
    folders = [(d, "LocIndependent") for d in cfg.radDistDirectories_LocIndependent]
    folders += [(d, "LocDependent") for d in cfg.radDistDirectories_LocDependent]
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
            if kind == "LocDependent" and float(info.get("injectionLocation", cfg.injectionLocation)) != float(cfg.injectionLocation):
                problems.append(f"{q.name}: injectionLocation {info.get('injectionLocation')} differs from the run config's {cfg.injectionLocation}")
        if problems:
            for msg in problems:
                err(f"{folder}: {msg}")
        else:
            ok(f"{folder}: {len(files)} radDist file(s); first {min(sample, len(files))} checked")
    return out
