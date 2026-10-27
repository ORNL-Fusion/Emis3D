# injector.py
"""
Shattered-pellet injectors, from the machine settings
(<machine>/<machine>_settings.yaml), in the same convention as the DREAM
runs' machine.yaml, plus the injector's toroidal angle:

    INJECTORS:
      SPI225:
        phi_deg: 225          # toroidal angle [deg], Emis3D coordinates (DIII-D: 360 - machine angle)
        shatter_R: 2.40       # where the pellet breaks [m]; every shard starts here
        shatter_Z: 0.0
        angle_deg: 0.0        # spray centre line in the poloidal plane: 0 = horizontally inward
                              #   (towards smaller R), + = upward; or aim_R / aim_Z instead;
                              #   with neither, the centre line points at the magnetic axis
        divergence_deg: 20.0  # full width of the spray cone [deg]
        speed: 200.0          # mean shard speed [m/s] (optional; for flight times)
        speed_spread: 0.15    # half-width of the shard speed spread, as a fraction of speed
        t_shatter_ms: null    # experiment time of the shatter (optional)
        verified: false       # false: every run that uses it says so
        note: placeholder geometry

A bare 'divergence' is refused: degrees and radians are both plausible
numbers for a cone width. The spray lies in the injector's poloidal plane
(no toroidal velocity), as in DREAM's prescribed-velocity SPI model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

INJECTOR_KEYS = {"phi_deg", "shatter_R", "shatter_Z", "angle_deg", "aim_R", "aim_Z", "divergence_deg", "speed",
                 "speed_spread", "t_shatter_ms", "seed", "verified", "note"}


@dataclass
class Injector:
    name: str
    phi_deg: float
    shatter_R: float
    shatter_Z: float
    angle_deg: float | None = None
    aim_R: float | None = None
    aim_Z: float | None = None
    divergence_deg: float = 20.0
    speed: float | None = None
    speed_spread: float = 0.15
    t_shatter_ms: float | None = None
    seed: int = 0
    verified: bool = False
    note: str = ""

    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls, name: str, entry: dict, where: str = "INJECTORS") -> "Injector":
        w = f"{where}.{name}"
        if not isinstance(entry, dict):
            raise ValueError(f"{w} must be a mapping of settings")
        if "divergence" in entry:
            raise ValueError(f"{w}: 'divergence' is ambiguous; give the full cone width in degrees as divergence_deg (e.g. 20.0)")
        unknown = sorted(set(entry) - INJECTOR_KEYS)
        if unknown:
            raise ValueError(f"{w}: unknown key(s) {unknown}; an injector takes {sorted(INJECTOR_KEYS)}")
        missing = [k for k in ("phi_deg", "shatter_R", "shatter_Z") if entry.get(k) is None]
        if missing:
            raise ValueError(f"{w} needs {missing}")
        if entry.get("angle_deg") is not None and (entry.get("aim_R") is not None or entry.get("aim_Z") is not None):
            raise ValueError(f"{w}: give angle_deg or aim_R/aim_Z, not both")
        for k in ("phi_deg", "shatter_R", "shatter_Z", "angle_deg", "aim_R", "aim_Z", "divergence_deg", "speed", "speed_spread", "t_shatter_ms"):
            v = entry.get(k)
            if v is not None and not isinstance(v, (int, float)):
                raise ValueError(f"{w}.{k} must be a number, got {v!r}")
        div = float(entry.get("divergence_deg", 20.0))
        if not 0.0 < div < 180.0:
            raise ValueError(f"{w}.divergence_deg must be between 0 and 180 (the full cone width in degrees), got {div}")
        if entry.get("speed") is not None and float(entry["speed"]) <= 0:
            raise ValueError(f"{w}.speed must be positive [m/s]")
        f = lambda k: None if entry.get(k) is None else float(entry[k])  # noqa: E731
        return cls(name=str(name), phi_deg=float(entry["phi_deg"]) % 360.0, shatter_R=float(entry["shatter_R"]), shatter_Z=float(entry["shatter_Z"]),
                   angle_deg=f("angle_deg"), aim_R=f("aim_R"), aim_Z=f("aim_Z"), divergence_deg=div, speed=f("speed"),
                   speed_spread=float(entry.get("speed_spread", 0.15)), t_shatter_ms=f("t_shatter_ms"), seed=int(entry.get("seed", 0)),
                   verified=bool(entry.get("verified", False)), note=str(entry.get("note", "")))

    # ------------------------------------------------------------------
    def direction(self, R_axis: float, Z_axis: float) -> float:
        """The spray centre line's direction in the poloidal plane, as an angle from +R [rad]."""
        if self.angle_deg is not None:
            return float(np.pi - np.radians(self.angle_deg))
        aim_R = R_axis if self.aim_R is None else self.aim_R
        aim_Z = Z_axis if self.aim_Z is None else self.aim_Z
        return float(np.arctan2(aim_Z - self.shatter_Z, aim_R - self.shatter_R))

    def centre_angle_deg(self, R_axis: float, Z_axis: float) -> float:
        """The centre line in the angle_deg convention (0 = horizontally inward, + = upward)."""
        return float(np.degrees(np.pi - self.direction(R_axis, Z_axis) + np.pi) % 360.0 - 180.0)

    def ray(self, R_axis: float, Z_axis: float, offset_deg: float = 0.0, length: float = 3.0, n: int = 2):
        """Points (R, Z) along a ray from the shatter point, offset_deg from the centre line."""
        th = self.direction(R_axis, Z_axis) + np.radians(offset_deg)
        s = np.linspace(0.0, length, n)
        return self.shatter_R + s * np.cos(th), self.shatter_Z + s * np.sin(th)

    def in_cone(self, R, Z, R_axis: float, Z_axis: float) -> np.ndarray:
        """True where (R, Z) lies inside the spray cone (in front of the shatter point)."""
        dR, dZ = np.asarray(R, dtype=float) - self.shatter_R, np.asarray(Z, dtype=float) - self.shatter_Z
        th = self.direction(R_axis, Z_axis)
        along = dR * np.cos(th) + dZ * np.sin(th)
        across = -dR * np.sin(th) + dZ * np.cos(th)
        return (along > 0) & (np.abs(np.arctan2(across, along)) <= np.radians(self.divergence_deg) / 2.0)

    def axis_miss(self, R_axis: float, Z_axis: float) -> float:
        """How far the centre line passes from the magnetic axis [m] (negative: the axis is behind the shatter point)."""
        th = self.direction(R_axis, Z_axis)
        dR, dZ = R_axis - self.shatter_R, Z_axis - self.shatter_Z
        along = dR * np.cos(th) + dZ * np.sin(th)
        miss = abs(-dR * np.sin(th) + dZ * np.cos(th))
        return float(miss if along >= 0 else -np.hypot(dR, dZ))

    def describe(self, R_axis: float | None = None, Z_axis: float | None = None) -> str:
        text = f"{self.name}: phi {self.phi_deg:g} deg, shatter ({self.shatter_R:.3f}, {self.shatter_Z:+.3f}) m, cone {self.divergence_deg:g} deg"
        if R_axis is not None and Z_axis is not None:
            text += f", centre line {self.centre_angle_deg(R_axis, Z_axis):+.1f} deg, passes {100 * self.axis_miss(R_axis, Z_axis):.1f} cm from the axis"
        if not self.verified:
            text += " (UNVERIFIED geometry" + (f": {self.note}" if self.note else "") + ")"
        return text

    def plot(self, ax, R_axis: float, Z_axis: float, label: str | None = None, color: str = "k") -> None:
        """The spray on a poloidal cross-section: shatter point, centre line, cone edges; the view limits are kept."""
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        half = self.divergence_deg / 2.0
        L = 3.0 * max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
        Re1, Ze1 = self.ray(R_axis, Z_axis, -half, L)
        Re2, Ze2 = self.ray(R_axis, Z_axis, +half, L)
        ax.fill([self.shatter_R, Re1[-1], Re2[-1]], [self.shatter_Z, Ze1[-1], Ze2[-1]], color=color, alpha=0.12, lw=0, zorder=4)
        ax.plot(Re1, Ze1, color=color, lw=0.8, ls="--", zorder=5)
        ax.plot(Re2, Ze2, color=color, lw=0.8, ls="--", zorder=5)
        Rc, Zc = self.ray(R_axis, Z_axis, 0.0, L)
        ax.plot(Rc, Zc, color=color, lw=1.2, zorder=5, label=label or f"{self.name} ({self.divergence_deg:g} deg cone)")
        ax.plot([self.shatter_R], [self.shatter_Z], marker="*", ms=12, color=color, mec="w", zorder=6)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


def load_injectors(settings: dict, where: str = "machine settings") -> dict:
    """{name: Injector} from a machine settings mapping's INJECTORS block (empty when there is none)."""
    block = (settings or {}).get("INJECTORS") or {}
    if not isinstance(block, dict):
        raise ValueError(f"{where}: INJECTORS must map injector names to their settings")
    return {name: Injector.from_settings(name, entry, f"{where}: INJECTORS") for name, entry in block.items()}


def machine_injectors(ws, machine: str) -> dict:
    """The injectors of a machine, from its settings file in the workspace (or the built-in machines)."""
    from emis3d.util import config_loader

    path = ws.machine_file(machine, f"{machine}_settings.yaml", required=False)
    if path is None:
        return {}
    return load_injectors(config_loader(path) or {}, str(path))
