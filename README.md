# Emis3D

This program will fit the synthetic radDist signals compared to experimental data. The
radDists and pre-processed SXR/Bolometer signals should be created prior to using this
routine. Each program requires config files to be run and some specific file organization,
but the general structure is this:


## Installation

Requires Python 3.11 or 3.12.

```bash
git clone https://github.com/ORNL-Fusion/Emis3D
cd Emis3D
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or, with [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
uv pip install -e .
```

### Python Compatibility

`cherab` and `raysect` ship Cython-compiled C extensions that don't build
on CPython 3.13+ — the `_PyLong_AsByteArray` API changed in that release.
Lift the cap once `cherab` ships a build compatible with newer Python.

### Troubleshooting

If `pip install -e .` fails with a `clang` / `_PyLong_AsByteArray` error,
the Python version is too new. Check with `python --version` and recreate
the venv with 3.12.


## Repository layout

```
src/emis3d/          the package (import emis3d)
  machines/MOCK/     the only built-in machine: synthetic, for tests and examples
  mock.py            generates MOCK; `python -m emis3d.mock` rewrites it
  workspace.py       where machines, data and results are found
  cli.py             the `emis3d` command
scripts/             make_radDists.py, run_emis3D.py
tools/DIII-D/        DIII-D data-preparation scripts (sxrSavers)
examples/  tests/  testingScripts/
```

Real machines (DIII-D, JET, SPARC, ...) are not part of the package. Each user
keeps the ones they need in their workspace's `machines/` folder.

## Workspace

Data and results live in a **workspace** folder of your choosing, outside the code:

```bash
emis3d init ~/projects/emis3d-runs --set-default   # create a workspace, remember it
emis3d paths                                       # where every location resolves
emis3d machines                                    # machines on the search path
emis3d selftest                                    # whole pipeline on MOCK (~10 s)
```

```
emis3d-runs/
  emis3d.yaml               locations (all optional; defaults written by `emis3d init`)
  machines/<machine>/       your machines, or files that override built-in ones
  equilibria/<machine>/     g-files (eqFileName is relative to this)
  measurements/<machine>/   processed SXR/bolometer HDF5 (dataFileName is relative to this)
  raddists/<machine>/       radDist libraries and their configs
  runs/<name>/run.yaml      one folder per analysis; results in runs/<name>/results/<run_id>/
  cache/
```

Emis3D finds the workspace from, in order: an explicit `workspace=` argument
or `--workspace`; the nearest `emis3d.yaml` above the run config; the
`EMIS3D_RUNS_DIR` environment variable; `default_workspace` in
`~/.config/emis3d/config.yaml`. Any location can point elsewhere (a shared
drive, scratch); paths may use `{tokamak}`, `{shot}`, `{run}`, `{run_dir}`,
`{run_id}`, `~` and `${ENV_VARS}`; quote values that contain braces. See
`src/emis3d/workspace.py` for the full rules.

```python
from emis3d.Emis3D import Emis3D
run = Emis3D(runConfigName="~/projects/emis3d-runs/runs/184407/run.yaml")
```

### Machines

A machine folder looks like this (see `src/emis3d/machines/MOCK/`):

```
<machine>_settings.yaml   volume, radii, wall file, injectors, CAD, list of cameras
inputs/
  <wall file>.txt
  CAD_stl_files/          optional
  sxrInfo/                one configuration file per SXR/bolometer camera
```

Machine files are searched file by file, workspace first and built-ins last,
so a single camera file in `machines/<machine>/inputs/sxrInfo/` adds or
overrides a camera.

### Moving from the old layout

Earlier versions kept machines in `tokamaks/` and data in `inputs/` inside the
repository. To move to a workspace:

```bash
emis3d init ~/projects/emis3d-runs --set-default
W=~/projects/emis3d-runs
# DIII-D's machine files, from the last commit that had them in the repository
git archive 4efabc6 tokamaks/DIII-D/DIII-D_settings.yaml tokamaks/DIII-D/inputs \
  | tar -x --strip-components=1 -C "$W/machines"
# data, for each machine you have
mv inputs/DIII-D/eqdsks    "$W/equilibria/DIII-D"
mv inputs/DIII-D/sxrData   "$W/measurements/DIII-D"
mv inputs/DIII-D/radDists  "$W/raddists/DIII-D"
# machines that were only on your disk (JET, SPARC, ...)
mv tokamaks/JET "$W/machines/"
```

Run configs move to `runs/<name>/` (for example `runs/184407/run.yaml`); their
`dataFileName` and `radDistDirectories_*` entries keep working unchanged.
`EMIS3D_ROOT` is no longer used.


The overall method to run this program is to:
------------------------------------------------------------------------------------------------
1. Create the radDists using the SXR/bolometer arrays that are availble for your shot
2. Grab, pre-process, and save the SXR data
3. Run Emis3D to find the best fit radDist for a given time segment.





Citations:
1. B. Stein-Lubrano et al 2024 Nucl. Fusion 64 036020
2. 