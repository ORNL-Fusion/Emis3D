# workspace.py
"""
Where Emis3D finds machines and data, and where it writes results.

Every location has a *role*. Each role is a path template, so data can live
wherever suits the user (a shared drive for equilibria, scratch for radDists,
a home directory for results) without editing any code.

    role          contents                                     default (new layout)
    ------------  -------------------------------------------  ------------------------------
    machines      tokamak settings, walls, CAD, camera configs  [machines]  + built-ins last
    equilibria    g-files                                       equilibria/{tokamak}
    measurements  processed SXR / bolometer HDF5                measurements/{tokamak}
    raddists      radDist libraries and their configs           raddists/{tokamak}
    runs          run configs                                   runs
    results       fits, figures, provenance                     {run_dir}/results/{run_id}
    cache         regenerable intermediate files                cache

Relative templates are relative to the workspace folder. Templates may use
{workspace}, {tokamak}, {shot}, {run}, {run_dir} and {run_id}, plus
'~' and environment variables such as ${SCRATCH}.

How the workspace is chosen (first match wins):
    1. an explicit argument                       Workspace.resolve(workspace=...)
    2. the nearest emis3d.yaml above a run config  (like git finding .git)
    3. the EMIS3D_RUNS_DIR environment variable
    4. default_workspace in ~/.config/emis3d/config.yaml
Without any of these only the built-in machines are available, and asking for
a data location raises an error that says how to create a workspace
(`emis3d init <folder>`).

Templates are merged role by role, lowest priority first: built-in defaults,
the user file (~/.config/emis3d/config.yaml), the workspace file
(emis3d.yaml, including its per-tokamak section), the run config's 'paths:'
block, then explicit overrides.

Machine files are looked up file by file: the user's machine folders are
searched in order and the machines shipped with the package come last. The
package ships only the synthetic MOCK machine (see emis3d.mock); real machines
such as DIII-D live in the workspace's machines/ folder.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import string
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
BUILTIN_MACHINES = PACKAGE_DIR / "machines"
REPO_ROOT = PACKAGE_DIR.parent.parent   # the source checkout, when there is one (src/emis3d)

WORKSPACE_FILE = "emis3d.yaml"
USER_CONFIG = Path("~/.config/emis3d/config.yaml")

ROLES = ("machines", "equilibria", "measurements", "raddists", "runs", "results", "cache")

DEFAULT_TEMPLATES = {
    "machines": ["machines"],
    "equilibria": "equilibria/{tokamak}",
    "measurements": "measurements/{tokamak}",
    "raddists": "raddists/{tokamak}",
    "runs": "runs",
    "results": "{run_dir}/results/{run_id}",
    "cache": "cache",
}

_WORKSPACE_KEYS = {"version", "paths", "tokamaks"}
_USER_KEYS = {"default_workspace", "paths"}


class WorkspaceError(RuntimeError):
    """Raised for configuration problems, with a message saying how to fix them."""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _read_yaml(path: Path, allowed: set, what: str) -> dict:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise WorkspaceError(f"{what} {path} must contain a mapping")
    unknown = set(data) - allowed
    if unknown:
        raise WorkspaceError(f"Unknown keys {sorted(unknown)} in {what} {path}; allowed: {sorted(allowed)}")
    paths = data.get("paths") or {}
    bad = set(paths) - set(ROLES)
    if bad:
        raise WorkspaceError(f"Unknown roles {sorted(bad)} in {path}; roles are {list(ROLES)}")
    return data


def _check_roles(paths: dict | None, where: str) -> dict:
    paths = dict(paths or {})
    bad = set(paths) - set(ROLES)
    if bad:
        raise WorkspaceError(f"Unknown roles {sorted(bad)} in {where}; roles are {list(ROLES)}")
    return paths


def _find_upwards(start: Path, name: str) -> Path | None:
    start = start if start.is_dir() else start.parent
    for d in [start, *start.parents]:
        if (d / name).is_file():
            return d
    return None


def _template_fields(template: str) -> set:
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


# ----------------------------------------------------------------------------
# Workspace
# ----------------------------------------------------------------------------


@dataclass
class Workspace:
    """Resolved locations. Create with Workspace.resolve(), not directly."""

    templates: dict
    fields: dict
    layout: str                                   # "workspace" or "unconfigured"
    sources: dict = field(default_factory=dict)   # role -> where its template came from
    tokamak_templates: dict = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: _dt.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # ------------------------------------------------------------------ build

    @classmethod
    def resolve(
        cls,
        workspace: str | os.PathLike | None = None,
        run_config: str | os.PathLike | None = None,
        run_paths: dict | None = None,
        overrides: dict | None = None,
        env: dict | None = None,
        user_config: str | os.PathLike | None = None,
    ) -> "Workspace":
        env = os.environ if env is None else env
        user_file = Path(os.path.expanduser(str(user_config or USER_CONFIG)))
        user = _read_yaml(user_file, _USER_KEYS, "user config") if user_file.is_file() else {}

        # --- 1-4: find the workspace folder
        ws_dir, how = None, None
        if workspace is not None:
            ws_dir, how = Path(workspace), "argument"
        elif run_config is not None and _find_upwards(Path(run_config).resolve(), WORKSPACE_FILE):
            ws_dir, how = _find_upwards(Path(run_config).resolve(), WORKSPACE_FILE), f"{WORKSPACE_FILE} above the run config"
        elif env.get("EMIS3D_RUNS_DIR"):
            ws_dir, how = Path(env["EMIS3D_RUNS_DIR"]), "EMIS3D_RUNS_DIR"
        elif user.get("default_workspace"):
            ws_dir, how = Path(user["default_workspace"]), f"default_workspace in {user_file}"

        if ws_dir is not None:
            ws_dir = Path(os.path.expandvars(os.path.expanduser(str(ws_dir)))).resolve()
            templates = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULT_TEMPLATES.items()}
            sources = {r: "default" for r in ROLES}
            fields = {"workspace": str(ws_dir)}
            layout = "workspace"
        else:
            templates = {"machines": []}
            sources = {"machines": "built-in only"}
            fields = {}
            layout = "unconfigured"
            how = "nothing configured"

        ws = cls(templates=templates, fields=fields, layout=layout, sources=sources)
        ws.fields["_how"] = how

        # --- merge templates, lowest priority first
        ws._merge(_check_roles(user.get("paths"), str(user_file)), f"user config {user_file}")
        if layout == "workspace":
            wfile = ws_dir / WORKSPACE_FILE
            if wfile.is_file():
                wdata = _read_yaml(wfile, _WORKSPACE_KEYS, "workspace file")
                ws._merge(_check_roles(wdata.get("paths"), str(wfile)), str(wfile))
                for tok, roles in (wdata.get("tokamaks") or {}).items():
                    ws.tokamak_templates[tok] = _check_roles(roles, f"{wfile} tokamaks.{tok}")
        ws._merge(_check_roles(run_paths, "run config paths"), "run config")
        ws._merge(_check_roles(overrides, "overrides"), "argument")
        return ws

    def _merge(self, paths: dict, source: str) -> None:
        for role, tmpl in paths.items():
            if role == "machines" and not isinstance(tmpl, list):
                tmpl = [tmpl]
            self.templates[role] = tmpl
            self.sources[role] = source
        if paths and self.layout == "unconfigured":
            # Explicit locations without a workspace folder: relative paths use the cwd
            self.layout = "workspace"
            self.fields.setdefault("workspace", str(Path.cwd()))

    # ------------------------------------------------------------------ paths

    def _expand(self, template: str, role: str, **fields) -> Path:
        # Environment variables and ~ first, so ${VAR} never reaches str.format
        expanded = os.path.expanduser(os.path.expandvars(template))
        unset = re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)", expanded)
        if unset:
            raise WorkspaceError(
                f"Role '{role}' template '{template}' uses unset environment variable(s) {unset}"
            )
        values = {**{k: v for k, v in self.fields.items() if not k.startswith("_")}, "run_id": self.run_id}
        values.update({k: v for k, v in fields.items() if v is not None})
        missing = _template_fields(expanded) - set(values)
        if missing:
            raise WorkspaceError(
                f"Role '{role}' template '{template}' needs {sorted(missing)}; pass them as arguments"
            )
        p = Path(expanded.format(**values))
        if not p.is_absolute():
            if "workspace" not in self.fields:
                raise self._unconfigured_error(role)
            p = Path(self.fields["workspace"]) / p
        return p

    def _unconfigured_error(self, role: str) -> WorkspaceError:
        return WorkspaceError(
            f"No location configured for '{role}'. Create a workspace with "
            "`emis3d init <folder>`, or set EMIS3D_RUNS_DIR to an existing one."
        )

    def path(self, role: str, tokamak: str | None = None, **fields) -> Path:
        """Folder for a role, e.g. ws.path('raddists', tokamak='DIII-D')."""
        if role not in ROLES or role == "machines":
            raise ValueError(f"path() takes one of {[r for r in ROLES if r != 'machines']}")
        template = self.tokamak_templates.get(tokamak or "", {}).get(role, self.templates.get(role))
        if template is None:
            raise self._unconfigured_error(role)
        return self._expand(template, role, tokamak=tokamak, **fields)

    def machine_roots(self, tokamak: str | None = None) -> list[Path]:
        """Machine folders in search order; the built-in machines are always last."""
        templates = self.tokamak_templates.get(tokamak or "", {}).get("machines", self.templates.get("machines", []))
        if not isinstance(templates, list):
            templates = [templates]
        roots = [self._expand(t, "machines", tokamak=tokamak) for t in templates]
        return [*roots, BUILTIN_MACHINES]

    def machine_file(self, tokamak: str, *parts: str, required: bool = True) -> Path | None:
        """First existing <root>/<tokamak>/<parts...> over the machine search path."""
        for root in self.machine_roots(tokamak):
            p = root.joinpath(tokamak, *parts)
            if p.exists():
                return p
        if required:
            looked = "\n  ".join(str(r / tokamak) for r in self.machine_roots(tokamak))
            raise FileNotFoundError(f"{'/'.join(parts)} not found for machine '{tokamak}'. Looked in:\n  {looked}")
        return None

    def has_machine(self, tokamak: str) -> bool:
        return self.machine_file(tokamak, f"{tokamak}_settings.yaml", required=False) is not None

    def machines(self) -> list[str]:
        """Names of every machine found on the search path."""
        names = set()
        for root in self.machine_roots():
            if root.is_dir():
                names.update(d.name for d in root.iterdir() if (d / f"{d.name}_settings.yaml").is_file())
        return sorted(names)

    def results_dir(self, create: bool = True, **fields) -> Path:
        """Results folder for one run; in the new layout a fresh {run_id} folder per run."""
        d = self.path("results", **fields)
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------- reporting

    def describe(self, tokamak: str | None = None, **fields) -> list[tuple[str, str, str]]:
        """(role, location or reason, source) for every role; used by `emis3d paths`."""
        rows = [("workspace", self.fields.get("workspace", "-"), self.fields.get("_how", ""))]
        for root in self.machine_roots(tokamak):
            src = "built-in" if root == BUILTIN_MACHINES else self.sources.get("machines", "")
            rows.append(("machines", str(root) + ("" if root.exists() else "  (missing)"), src))
        for role in ROLES:
            if role == "machines":
                continue
            try:
                p = self.path(role, tokamak=tokamak, **fields)
                where = str(p) + ("" if p.exists() else "  (missing)")
            except WorkspaceError as e:
                where = str(e).split(";")[0]
            rows.append((role, where, self.sources.get(role, "")))
        return rows

    def export_env(self) -> None:
        """Make child processes (e.g. a process pool) resolve the same workspace."""
        if self.layout == "workspace":
            os.environ["EMIS3D_RUNS_DIR"] = self.fields["workspace"]

    def provenance(self, run_config: str | os.PathLike | None = None, **fields) -> dict:
        """Record of code version and resolved locations, saved beside results."""
        try:
            from importlib.metadata import version
            pkg_version = version("emis3d")
        except Exception:
            pkg_version = "unknown"
        commit, dirty = None, None
        if (REPO_ROOT / ".git").exists():
            try:
                commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                        capture_output=True, text=True, timeout=5).stdout.strip() or None
                dirty = bool(subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                                            capture_output=True, text=True, timeout=5).stdout.strip())
            except Exception:
                pass
        return {
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "emis3d_version": pkg_version,
            "git_commit": commit,
            "git_uncommitted_changes": dirty,
            "layout": self.layout,
            "run_id": self.run_id,
            "run_config": str(run_config) if run_config else None,
            "locations": {role: where for role, where, _ in self.describe(**fields)},
        }


# ----------------------------------------------------------------------------
# Process-wide default
# ----------------------------------------------------------------------------

_CURRENT: Workspace | None = None


def get_workspace() -> Workspace:
    """The workspace used when none is passed explicitly (resolved lazily, once)."""
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = Workspace.resolve()
    return _CURRENT


def set_workspace(ws: Workspace | None) -> None:
    """Set (or with None, reset) the process-wide workspace."""
    global _CURRENT
    _CURRENT = ws
