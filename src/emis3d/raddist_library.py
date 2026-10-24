# raddist_library.py
"""
A radDist library config: one file describing every radDist type to build
for a shot, with shared settings at the top and one block per type.

    tokamakName: JET
    shot: 95709
    folder: 95709                    # under the workspace's raddists/<tokamak>
    injectionLocation: 11            # deg
    units: Brightness
    eqFileName: g_JET_p95707_t50.544800
    numBins: 100
    workers: 7
    rays: {pixelSamples: 1000, seed: 0, integrationStep: 0.01, method: compiled}
        # or method: matrix, matrix: medium  (a geometry matrix from emis3d build-matrix; no ray tracing)
    grid: {rLimits: [], zLimits: [], NumR: 10, NumZ: 20}

    types:
      elongatedRing:                 # built into <folder>/elongatedRing
        distType: ElongatedRing
        sigma_R: [0.2, 0.3]          # a list is swept; every distinct combination is built
        sigma_z: [0.4]
        rotationAngle: [0, 30, 60]
      helical:
        distType: Helical
        sigma_R: [0.05]
        sigmaKernel: 0.01
        numFieldLines: 500
        grid: {rLimits: [2.4, 3.4], zLimits: [0.8, 1.6], NumR: 16, NumZ: 16}   # overrides the shared grid
      nimrod:
        distType: NIMROD
        files: [nimrod_prad.json]    # relative to this file's folder, or absolute
        timesteps: [40, 41]

Every key is checked: an unknown key, a key the type does not use or a
missing required key is an error naming the type and the key. The radDist
classes keep their own setting names; radDist_config() translates.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

from emis3d.util import config_loader

# --- Settings shared by every type (any of OVERRIDABLE can be set again inside a type)
SHARED_KEYS = {"tokamakName", "shot", "folder", "injectionLocation", "units", "eqFileName", "numBins", "workers", "rays", "grid", "types"}
OVERRIDABLE = {"injectionLocation", "units", "eqFileName", "numBins", "workers", "rays", "grid"}
RAYS_KEYS = {"pixelSamples", "seed", "integrationStep", "method", "gridSpacing", "matrix", "supersample"}
RAY_METHODS = ("compiled", "python", "matrix")
GRID_KEYS = {"rLimits", "zLimits", "NumR", "NumZ"}
UNITS = ("Power", "Radiance", "Brightness")

# --- What each type takes: required keys, optional keys with defaults. Shape
# keys may be lists, which are swept.
TYPES = {
    "ElongatedRing": {"required": ("sigma_R", "sigma_z"), "optional": {"rotationAngle": 0.0}},
    "Helical": {"required": ("sigma_R", "sigmaKernel", "numFieldLines"), "optional": {"numTransits": 1}},
    "HelicalRing": {"required": ("sigma_R", "sigma_z"), "optional": {"numTransits": 1}},
    "SquareTube": {"required": ("dR", "dz"), "optional": {}},
    "NIMROD": {"required": ("files", "timesteps"), "optional": {}},
    "M3DC1": {"required": ("files", "timesteps"), "optional": {}},
}
SIMULATION_TYPES = ("NIMROD", "M3DC1")
NEEDS_EQUILIBRIUM = ("Helical", "HelicalRing")

# Names the radDist classes use for some settings
CLASS_NAMES = {"numTransits": "numTransists"}

DEFAULT_RAYS = {"pixelSamples": 1000, "seed": 0, "integrationStep": 0.01, "method": "compiled"}


@dataclass
class RadDistType:
    """One type block, merged with the shared settings."""

    name: str
    distType: str
    settings: dict  # shared settings, overridden by the block's own
    shape: dict  # this type's shape settings (values or lists)
    enabled: bool = True

    @property
    def is_simulation(self) -> bool:
        return self.distType in SIMULATION_TYPES


@dataclass
class RadDistLibrary:
    path: Path
    tokamakName: str
    folder: str
    shot: int | None
    types: list[RadDistType] = field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path) -> "RadDistLibrary":
        path = Path(path).expanduser()
        raw = config_loader(path)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping of settings")
        return cls.from_dict(raw, path)

    @classmethod
    def from_dict(cls, raw: dict, path: Path | str = "<dict>") -> "RadDistLibrary":
        path = Path(path)
        errors: list[str] = []

        unknown = sorted(set(raw) - SHARED_KEYS)
        if unknown:
            errors.append(f"unknown setting(s) {unknown}; the top level takes {sorted(SHARED_KEYS)}")
        for key in ("tokamakName", "injectionLocation", "types"):
            if key not in raw:
                errors.append(f"missing {key!r}")
        if not isinstance(raw.get("types", {}), dict) or not raw.get("types"):
            errors.append("'types' must name at least one type, e.g. types: {elongatedRing: {distType: ElongatedRing, ...}}")
        folder = raw.get("folder", raw.get("shot"))
        if folder is None:
            errors.append("missing 'folder' (or 'shot', used as the folder)")
        shared = {k: raw[k] for k in OVERRIDABLE if k in raw}
        errors += _check_shared(shared, "")

        types = []
        for name, block in (raw.get("types") or {}).items() if isinstance(raw.get("types"), dict) else []:
            t, errs = _parse_type(str(name), block, shared)
            errors += errs
            if t is not None:
                types.append(t)

        if errors:
            raise ValueError(f"radDist library {path}:\n  - " + "\n  - ".join(errors))
        return cls(path=path, tokamakName=str(raw["tokamakName"]), folder=str(folder), shot=raw.get("shot"), types=types)

    # ------------------------------------------------------------------
    def select(self, only=None) -> list[RadDistType]:
        """The enabled types, or those named in `only` (enabled or not)."""
        if not only:
            return [t for t in self.types if t.enabled]
        names = {t.name for t in self.types}
        missing = sorted(set(only) - names)
        if missing:
            raise ValueError(f"no type(s) {missing} in {self.path}; it has {sorted(names)}")
        return [t for t in self.types if t.name in set(only)]

    def shape_sets(self, t: RadDistType) -> list[dict]:
        """
        Every distinct shape to build for a type, as the radDist classes'
        settings. Lists are swept; elongated rings skip duplicates (circles at
        several angles, angles 180 deg apart) via util_radDist.shape_parameter_sets.
        """
        if t.is_simulation:
            base = self.path.parent
            files = [str(p if Path(p).is_absolute() else (base / p).resolve()) for p in _as_list(t.shape["files"])]
            return [{"sourceFile": f, "timestep": int(ts)} for f in files for ts in _as_list(t.shape["timesteps"])]
        if t.distType == "ElongatedRing":
            import emis3d.util_radDist as Util_radDist

            rings = Util_radDist.shape_parameter_sets("ElongatedRing", _as_list(t.shape["sigma_R"]), _as_list(t.shape["sigma_z"]),
                                                      _as_list(t.shape["rotationAngle"]))
            return [{k: (float(v) if v is not None else v) for k, v in r.items()} for r in rings]
        keys = list(t.shape)
        combos = itertools.product(*(_as_list(t.shape[k]) for k in keys))
        return [{CLASS_NAMES.get(k, k): v for k, v in zip(keys, combo, strict=True)} for combo in combos]

    def radDist_config(self, t: RadDistType, shape: dict) -> dict:
        """The settings one radDist of this type is built with (the classes' names)."""
        s = t.settings
        rays = {**DEFAULT_RAYS, **(s.get("rays") or {})}
        cfg = {
            "distType": t.distType,
            "tokamakName": self.tokamakName,
            "shotNumber": self.shot,
            "units": s.get("units", "Brightness"),
            "injectionLocation": float(s["injectionLocation"]),
            "numBins": int(s.get("numBins", 100)),
            "eqFileName": s.get("eqFileName"),
            "integrationStep": float(rays["integrationStep"]),
            "BOLOMETER_PROPS": {"pixelSamples": int(rays["pixelSamples"]), "seed": rays.get("seed")},
            "rayMethod": rays["method"],
            "saveRunsDirectoryName": f"{self.folder}/{t.name}",
        }
        if rays.get("gridSpacing"):
            cfg["gridSpacing"] = float(rays["gridSpacing"])
        if rays["method"] == "matrix":
            cfg["matrixName"] = str(rays.get("matrix", "medium"))  # a matrix in the cache (emis3d build-matrix)
            if rays.get("supersample"):
                cfg["supersample"] = int(rays["supersample"])
        cfg.update(shape)
        return cfg

    def folder_for(self, t: RadDistType) -> str:
        """The type's folder, relative to raddists/<tokamak>: what a run config lists."""
        return f"{self.folder}/{t.name}"


# ---------------------------------------------------------------------------


def _as_list(v) -> list:
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _check_shared(s: dict, where: str) -> list[str]:
    errors = []
    if "units" in s and s["units"] not in UNITS:
        errors.append(f"{where}units must be one of {UNITS}, got {s['units']!r}")
    if "rays" in s:
        if not isinstance(s["rays"], dict):
            errors.append(f"{where}rays must be a mapping, e.g. {{pixelSamples: 1000, seed: 0}}")
        else:
            bad = sorted(set(s["rays"]) - RAYS_KEYS)
            if bad:
                errors.append(f"{where}rays: unknown key(s) {bad}; it takes {sorted(RAYS_KEYS)}")
            if s["rays"].get("method", "compiled") not in RAY_METHODS:
                errors.append(f"{where}rays.method must be one of {RAY_METHODS}, got {s['rays']['method']!r}")
            if "gridSpacing" in s["rays"] and not (isinstance(s["rays"]["gridSpacing"], (int, float)) and s["rays"]["gridSpacing"] > 0):
                errors.append(f"{where}rays.gridSpacing must be a positive length in metres")
            if "supersample" in s["rays"] and not (isinstance(s["rays"]["supersample"], int) and 1 <= s["rays"]["supersample"] <= 6):
                errors.append(f"{where}rays.supersample must be a whole number from 1 to 6")
    if "grid" in s:
        if not isinstance(s["grid"], dict):
            errors.append(f"{where}grid must be a mapping, e.g. {{rLimits: [], zLimits: [], NumR: 10, NumZ: 20}}")
        else:
            bad = sorted(set(s["grid"]) - GRID_KEYS)
            if bad:
                errors.append(f"{where}grid: unknown key(s) {bad}; it takes {sorted(GRID_KEYS)}")
            missing = [k for k in ("NumR", "NumZ") if k not in s["grid"]]
            if missing:
                errors.append(f"{where}grid is missing {missing}")
    if "workers" in s and (not isinstance(s["workers"], int) or s["workers"] < 1):
        errors.append(f"{where}workers must be a whole number >= 1")
    return errors


def _parse_type(name: str, block, shared: dict) -> tuple[RadDistType | None, list[str]]:
    where = f"types.{name}: "
    if not isinstance(block, dict):
        return None, [f"{where}expected a mapping of settings"]
    dist = block.get("distType")
    if dist not in TYPES:
        return None, [f"{where}distType must be one of {sorted(TYPES)}, got {dist!r}"]
    spec = TYPES[dist]
    shape_keys = set(spec["required"]) | set(spec["optional"])

    errors = []
    unknown = sorted(set(block) - shape_keys - OVERRIDABLE - {"distType", "enabled"})
    if unknown:
        errors.append(f"{where}unknown key(s) {unknown} for {dist}; it takes {sorted(shape_keys)} "
                      f"(and may override {sorted(OVERRIDABLE)})")
    missing = [k for k in spec["required"] if k not in block]
    if missing:
        errors.append(f"{where}{dist} needs {missing}")

    settings = {**shared, **{k: block[k] for k in OVERRIDABLE if k in block}}
    errors += _check_shared({k: block[k] for k in OVERRIDABLE if k in block}, where)
    if "injectionLocation" not in settings:
        errors.append(f"{where}no injectionLocation")
    if dist not in SIMULATION_TYPES and "grid" not in settings:
        errors.append(f"{where}no grid of start positions (set 'grid' at the top or in the type)")
    if dist in NEEDS_EQUILIBRIUM and not settings.get("eqFileName"):
        errors.append(f"{where}{dist} needs an eqFileName (an equilibrium to trace field lines in)")

    # In the type's own order (required, then optional), so reports read consistently
    order = list(spec["required"]) + list(spec["optional"])
    shape = {k: block.get(k, spec["optional"].get(k)) for k in order if k in block or k in spec["optional"]}
    for k, v in shape.items():
        if isinstance(v, (list, tuple)) and len(v) == 0:
            errors.append(f"{where}{k} is an empty list")
    if errors:
        return None, errors
    return RadDistType(name=name, distType=dist, settings=settings, shape=shape, enabled=bool(block.get("enabled", True))), []
