# cli.py
"""
The ``emis3d`` command.

    emis3d init <folder> [--set-default]   create a workspace
    emis3d paths [--config RUN] [--workspace DIR] [--tokamak T] [--shot N]
    emis3d machines [--workspace DIR]      list machines on the search path
    emis3d selftest [--keep DIR]           run the whole pipeline on the MOCK machine
    emis3d calibrate RUN --times T [T ...] [--reference CAMERA] [--output FILE]
                                           cross-calibrate cameras (emis3d.calibration)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from emis3d.workspace import (
    DEFAULT_TEMPLATES,
    USER_CONFIG,
    WORKSPACE_FILE,
    Workspace,
    WorkspaceError,
)

WORKSPACE_TEMPLATE = """\
# Emis3D workspace settings. Every key is optional; the values below are the
# defaults. Relative paths are relative to this folder. Templates may use
# {{tokamak}}, {{shot}}, {{run}}, {{run_dir}} and {{run_id}}, plus ~ and ${{ENV_VARS}}.
# Quote any value that contains braces. Run `emis3d paths` to see where each
# location resolves.
version: 1

paths:
  machines: ["machines"]                  # searched first; built-in machines come last
  equilibria: "{equilibria}"             # g-files
  measurements: "{measurements}"         # processed SXR/bolometer HDF5; dataFileName is relative to it
  raddists: "{raddists}"                 # radDist libraries and their configs
  runs: "{runs}"                                  # run configs
  results: "{results}"         # one new folder per run
  cache: "{cache}"                                # regenerable files; safe to delete

# Per-machine exceptions, e.g. JET data on a shared drive:
# tokamaks:
#   JET:
#     measurements: "/work/jet/sxr"
#     equilibria: "/work/jet/eqdsk"
"""

RUN_TEMPLATE = """\
# Example run configuration. Copy this folder to runs/<shot> and edit.
# File names below are relative to the workspace locations shown by `emis3d paths`.
tokamakName: MOCK                         # the built-in synthetic machine; e.g. DIII-D once it is in machines/
shot: 1
dt: 0.3                                   # averaging half-width around each evaluation time [ms]
injectionLocation: 225                    # [deg, Emis3D convention]

radDistDirectories_LocIndependent: []     # folders under <raddists>
radDistDirectories_LocDependent: []

BOLOMETERS:
  POL_A:
    dataFileName: 1/POL_A.h5              # under <measurements>
    scalingFactor: 1.0

scale_def: "exponential"
error_definition: "exponential"
enable_dphi_scaling: True
vary_peak_rad_location: False
"""

GITIGNORE = ".venv/\nequilibria/\nmeasurements/\nraddists/\ncache/\nruns/*/results/\n"


def cmd_init(args) -> int:
    root = Path(args.folder).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    wfile = root / WORKSPACE_FILE
    if wfile.exists() and not args.force:
        print(f"{wfile} already exists; use --force to overwrite it", file=sys.stderr)
        return 1

    for d in ("machines", "equilibria", "measurements", "raddists", "runs/example", "cache"):
        (root / d).mkdir(parents=True, exist_ok=True)
    wfile.write_text(WORKSPACE_TEMPLATE.format(**DEFAULT_TEMPLATES))
    example = root / "runs" / "example" / "run.yaml"
    if not example.exists():
        example.write_text(RUN_TEMPLATE)
    gi = root / ".gitignore"
    if not gi.exists():
        gi.write_text(GITIGNORE)

    if args.set_default:
        ufile = USER_CONFIG.expanduser()
        ufile.parent.mkdir(parents=True, exist_ok=True)
        data = (yaml.safe_load(ufile.read_text()) or {}) if ufile.exists() else {}
        data["default_workspace"] = str(root)
        ufile.write_text(yaml.safe_dump(data, sort_keys=False))
        print(f"Default workspace set in {ufile}")

    print(f"Workspace created: {root}")
    print("Next: put data in its folders (or point emis3d.yaml elsewhere), then run `emis3d paths`.")
    return 0


def _resolve(args) -> Workspace:
    run_paths = None
    if getattr(args, "config", None):
        with open(Path(args.config).expanduser()) as fh:
            run = yaml.safe_load(fh) or {}
        run_paths = run.get("paths")
        args.tokamak = args.tokamak or run.get("tokamakName")
        args.shot = args.shot or run.get("shot")
    return Workspace.resolve(workspace=args.workspace, run_config=args.config, run_paths=run_paths)


def cmd_paths(args) -> int:
    ws = _resolve(args)
    fields = {}
    if args.shot is not None:
        fields["shot"] = args.shot
    if args.config:
        cfg = Path(args.config).expanduser().resolve()
        fields.update(run=cfg.stem, run_dir=str(cfg.parent))
    print(f"layout: {ws.layout}" + (f"   tokamak: {args.tokamak}" if args.tokamak else ""))
    rows = ws.describe(tokamak=args.tokamak, **fields)
    width = max(len(r[0]) for r in rows)
    for role, where, source in rows:
        print(f"{role:<{width}}  {where}" + (f"   [{source}]" if source else ""))
    return 0


def cmd_machines(args) -> int:
    ws = Workspace.resolve(workspace=args.workspace)
    for name in ws.machines():
        settings = ws.machine_file(name, f"{name}_settings.yaml")
        print(f"{name:<12} {settings}")
    return 0


def cmd_selftest(args) -> int:
    from emis3d.mock import selftest

    summary = selftest(workdir=args.keep, pixel_samples=args.pixel_samples)
    if args.keep:
        print(f"Results kept in {summary['results']}")
    return 0


def cmd_calibrate(args) -> int:
    from emis3d.calibration import cross_calibrate, parse_times

    result = cross_calibrate(
        args.config, parse_times(args.times), reference=args.reference, workspace=args.workspace
    )
    print()
    print(result.table())
    block = result.yaml_block()
    print()
    print(block)
    if args.output:
        Path(args.output).expanduser().write_text(block)
        print(f"Written to {args.output}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="emis3d", description="Emis3D command-line tools")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("init", help="create a workspace folder")
    pi.add_argument("folder")
    pi.add_argument("--set-default", action="store_true",
                    help=f"record it as default_workspace in {USER_CONFIG}")
    pi.add_argument("--force", action="store_true", help="overwrite an existing emis3d.yaml")
    pi.set_defaults(func=cmd_init)

    pp = sub.add_parser("paths", help="show where every location resolves")
    pp.add_argument("--config", help="run config whose paths and fields to use")
    pp.add_argument("--workspace")
    pp.add_argument("--tokamak")
    pp.add_argument("--shot")
    pp.set_defaults(func=cmd_paths)

    pm = sub.add_parser("machines", help="list machines on the search path")
    pm.add_argument("--workspace")
    pm.set_defaults(func=cmd_machines)

    ps = sub.add_parser("selftest", help="run the whole pipeline on the MOCK machine")
    ps.add_argument("--keep", help="folder to keep the self-test workspace in (default: temporary)")
    ps.add_argument("--pixel-samples", type=int, default=50)
    ps.set_defaults(func=cmd_selftest)

    pc = sub.add_parser("calibrate", help="cross-calibrate cameras against an axisymmetric radDist")
    pc.add_argument("config", help="run config whose radDist folders hold axisymmetric radDists")
    pc.add_argument("--times", nargs="+", required=True,
                    help="evaluation times, e.g. 2124 or 2120:2128:0.5 (start:stop:step, inclusive)")
    pc.add_argument("--reference", help="camera the others are scaled to (default: largest amplitude)")
    pc.add_argument("--workspace")
    pc.add_argument("--output", help="also write the BOLOMETERS block to this file")
    pc.set_defaults(func=cmd_calibrate)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (WorkspaceError, FileNotFoundError, ValueError) as e:
        print(f"emis3d: {e}", file=sys.stderr)
        return 2
    except AssertionError as e:
        print(f"emis3d selftest FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
