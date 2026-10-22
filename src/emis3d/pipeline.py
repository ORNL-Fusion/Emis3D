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


def make_raddists(config, tokamak: str | None = None, workspace=None, only=None, workers: int | None = None, dry_run: bool = False) -> dict:
    """
    Build the radDists of a library config (emis3d.raddist_library), every
    enabled type or only those named in `only`.

    config  : a path, or a name in the workspace's raddists/<tokamak> folder
    workers : radDists built in parallel (default: the config's workers, else 1)
    dry_run : only count what would be built

    Each type is built into raddists/<tokamak>/<folder>/<type name>; after a
    build, manifest.yaml in raddists/<tokamak>/<folder> records the config,
    the counts and the code version. Returns
    {"types": [{name, distType, shapes, start_positions, radDists, folder}], "radDists": total}.
    """
    import yaml

    import emis3d.util_radDist as Util_radDist
    from emis3d.raddist_library import RadDistLibrary
    from emis3d.workspace import Workspace, set_workspace

    ws = workspace if isinstance(workspace, Workspace) else Workspace.resolve(workspace=workspace)
    set_workspace(ws)
    ws.export_env()  # worker processes resolve the same workspace

    path = _library_path(config, tokamak, ws)
    lib = RadDistLibrary.load(path)
    if tokamak and tokamak != lib.tokamakName:
        raise ValueError(f"{path} is for {lib.tokamakName}, not {tokamak}")
    selected = lib.select(only)

    builders = {
        "ElongatedRing": Util_radDist.radDist_ElongatedRing_parallel,
        "Helical": Util_radDist.radDist_Helical_parallel,
        "HelicalRing": Util_radDist.radDist_HelicalRing_parallel,
        "SquareTube": Util_radDist.radDist_SquareTube_parallel,
        "NIMROD": Util_radDist.radDist_NIMROD_parallel,
        "M3DC1": Util_radDist.radDist_M3DC1_parallel,
    }
    root = ws.path("raddists", tokamak=lib.tokamakName)

    plans = []
    for t in selected:
        shapes = lib.shape_sets(t)
        if t.is_simulation:
            jobs = [(shape["sourceFile"], shape["timestep"], lib.radDist_config(t, shape)) for shape in shapes]
            positions = 1
        else:
            rz = start_positions(lib, t, ws)
            jobs = [(val, lib.radDist_config(t, shape)) for shape in shapes for val in rz]
            positions = len(rz)
        plans.append((t, jobs, {"name": t.name, "distType": t.distType, "shapes": len(shapes), "start_positions": positions,
                                "radDists": len(jobs), "folder": str(root / lib.folder_for(t))}))

    summary = {"types": [p[2] for p in plans], "radDists": sum(p[2]["radDists"] for p in plans)}
    if dry_run:
        return summary

    for t, jobs, _ in plans:
        n_workers = max(1, int(workers or t.settings.get("workers") or 1))
        logger.info("Building %d %s radDists (%s) with %d worker(s)", len(jobs), t.distType, t.name, n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for _ in executor.map(builders[t.distType], jobs):  # consume, so worker errors surface
                pass

    manifest = {
        "config": str(path.resolve()),
        "built": summary["types"],
        "provenance": ws.provenance(),
        "library": yaml.safe_load(path.read_text()),
    }
    out = root / lib.folder / "manifest.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return summary


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
