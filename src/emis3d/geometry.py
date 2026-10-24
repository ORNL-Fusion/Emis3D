# geometry.py
"""
The geometry matrix G: a linear forward model for the cameras.

A channel's signal is the volume integral of the emissivity weighted by the
channel's view, so on a voxel grid it is a matrix-vector product,

    power_i = sum_j G_ij eps_j / (4 pi)       [W, with eps in W/m^3]

G depends only on the cameras and the vessel. It is built once per machine
and camera (Cherab's RayTransferCylinder: every voxel is given its own
spectral bin, so one observe per channel returns that channel's whole row),
stored sparse in the workspace's cache, and reused: every candidate
emissivity afterwards costs one matrix-vector product instead of a ray trace.

G is built in the same world Emis3D ray-traces in (Tokamak, mode "Build": an
absorbing cylindrical annulus around the vessel), and the wall shape is
applied as a mask on the voxel emissivity, as calc_emissivity does for the
ray tracer, so G reproduces bolos_observe to within Monte Carlo noise.
Signals are converted exactly as bolos_observe converts a foil's power:
radiance = power / etendue, brightness = 4 pi radiance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# Voxel sizes [m] and toroidal sectors of the resolution presets
PRESETS = {
    "low": {"dR": 0.05, "dz": 0.05, "n_phi": 72},
    "medium": {"dR": 0.025, "dz": 0.025, "n_phi": 144},
    "high": {"dR": 0.015, "dz": 0.015, "n_phi": 216},
}
DEFAULT_SAMPLES = {"low": 20_000, "medium": 60_000, "high": 150_000}


@dataclass(frozen=True)
class GridSpec:
    """A cylindrical (R, phi, z) voxel grid: edges in R and z, n_phi equal sectors over a full turn."""

    r_edges: np.ndarray
    z_edges: np.ndarray
    n_phi: int

    @classmethod
    def from_wall(cls, wall: dict, resolution: str | dict = "medium", padding: float = 0.02) -> "GridSpec":
        """A grid covering the wall's extent (plus padding) at a preset or {dR, dz, n_phi}."""
        p = PRESETS[resolution] if isinstance(resolution, str) else dict(resolution)
        r0, r1 = wall["minr"] - padding, wall["maxr"] + padding
        z0, z1 = wall["minz"] - padding, wall["maxz"] + padding
        n_r = max(2, int(np.ceil((r1 - r0) / p["dR"])))
        n_z = max(2, int(np.ceil((z1 - z0) / p["dz"])))
        return cls(np.linspace(r0, r1, n_r + 1), np.linspace(z0, z1, n_z + 1), int(p["n_phi"]))

    @property
    def n_r(self) -> int:
        return self.r_edges.size - 1

    @property
    def n_z(self) -> int:
        return self.z_edges.size - 1

    @property
    def n_voxels(self) -> int:
        return self.n_r * self.n_phi * self.n_z

    @property
    def r_centers(self) -> np.ndarray:
        return 0.5 * (self.r_edges[:-1] + self.r_edges[1:])

    @property
    def z_centers(self) -> np.ndarray:
        return 0.5 * (self.z_edges[:-1] + self.z_edges[1:])

    @property
    def phi_centers(self) -> np.ndarray:
        return (np.arange(self.n_phi) + 0.5) * 2.0 * np.pi / self.n_phi

    def describe(self) -> str:
        dR = np.diff(self.r_edges).mean() * 100
        dz = np.diff(self.z_edges).mean() * 100
        return f"{self.n_r} x {self.n_phi} x {self.n_z} = {self.n_voxels:,} voxels ({dR:.1f} cm x {360 / self.n_phi:.2f} deg x {dz:.1f} cm)"


@dataclass
class GeometryMatrix:
    """
    G for one machine and a set of cameras, in Emis3D's brightness units:
    brightness_i = rows[i] . eps  [W/m^2] for eps in W/m^3 on the voxel grid
    (rows = G_power / etendue_i, with G_power in m^3 sr).

    Voxel j is grid index (i_r, i_phi, i_z) flattened in that order
    (numpy C order); `inside_wall` marks the voxels inside the wall polygon.
    """

    machine: str
    grid: GridSpec
    rows: sp.csr_matrix  # (channels, voxels)
    cameras: list[str]  # camera name of each row's group, in order
    channels: list[str]  # channel tag per row
    camera_of_row: np.ndarray  # index into `cameras` per row
    etendues: np.ndarray  # per row [m^2 sr]
    inside_wall: np.ndarray  # (voxels,) bool
    samples: int
    provenance: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def eps_index(self, i_r, i_phi, i_z):
        return (np.asarray(i_r) * self.grid.n_phi + np.asarray(i_phi)) * self.grid.n_z + np.asarray(i_z)

    def seen_voxels(self) -> np.ndarray:
        """Indices of the voxels any channel sees (the only ones an emissivity needs values at)."""
        return np.unique(self.rows.indices)

    def signals(self, eps: np.ndarray, units: str = "Brightness") -> np.ndarray:
        """Every channel's signal for an emissivity eps [W/m^3] on the grid (length n_voxels)."""
        b = self.rows @ np.asarray(eps, dtype=float)
        if units == "Brightness":
            return b
        if units == "Radiance":
            return b / (4.0 * np.pi)
        if units == "Power":
            return b * self.etendues / (4.0 * np.pi)
        raise ValueError(f"units must be Power, Radiance or Brightness, got {units!r}")

    def emissivity_on_grid(self, radDist, emissionName: str, only_seen: bool = True) -> np.ndarray:
        """
        A radDist's emissivity at the voxel centres (zero outside the wall),
        one toroidal plane at a time (every radDist type takes a single
        angle per call). With only_seen, only the voxels some channel sees
        are evaluated; the rest are zero.
        """
        g = self.grid
        eps = np.zeros(g.n_voxels)
        RR, ZZ = np.meshgrid(g.r_centers, g.z_centers, indexing="ij")
        for k, phi in enumerate(g.phi_centers):
            idx = self.eps_index(np.arange(g.n_r)[:, None], k, np.arange(g.n_z)[None, :]).ravel()
            use = self.inside_wall[idx]
            if only_seen:
                use &= np.isin(idx, self._seen, assume_unique=False)
            if not use.any():
                continue
            vals = radDist.calc_emissivity(RR.ravel()[use], ZZ.ravel()[use], phi=np.array([phi]), emissionName=emissionName)[emissionName]
            eps[idx[use]] = np.asarray(vals, dtype=float)
        return eps

    def observe(self, radDist, units: str | None = None) -> dict:
        """
        Signals of every emission of a radDist, per camera and channel, as
        bolos_observe stores them: {emissionName: {camera: [signal per channel]}}.
        """
        units = units or radDist.info.get("units", "Brightness")
        out = {}
        for name in radDist.info["emissionNames"]:
            b = self.signals(self.emissivity_on_grid(radDist, name), units)
            out[name] = {cam: b[self.camera_of_row == i].tolist() for i, cam in enumerate(self.cameras)}
        return out

    def __post_init__(self):
        self._seen = self.seen_voxels()

    # ------------------------------------------------------------------
    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            machine=self.machine,
            r_edges=self.grid.r_edges,
            z_edges=self.grid.z_edges,
            n_phi=self.grid.n_phi,
            data=self.rows.data,
            indices=self.rows.indices,
            indptr=self.rows.indptr,
            shape=np.array(self.rows.shape),
            cameras=np.array(self.cameras),
            channels=np.array(self.channels),
            camera_of_row=self.camera_of_row,
            etendues=self.etendues,
            inside_wall=self.inside_wall,
            samples=self.samples,
            provenance=np.array([str(self.provenance)]),
        )
        return path

    @classmethod
    def load(cls, path) -> "GeometryMatrix":
        import ast

        with np.load(path, allow_pickle=False) as f:
            grid = GridSpec(f["r_edges"], f["z_edges"], int(f["n_phi"]))
            rows = sp.csr_matrix((f["data"], f["indices"], f["indptr"]), shape=tuple(f["shape"]))
            return cls(
                machine=str(f["machine"]),
                grid=grid,
                rows=rows,
                cameras=[str(c) for c in f["cameras"]],
                channels=[str(c) for c in f["channels"]],
                camera_of_row=f["camera_of_row"],
                etendues=f["etendues"],
                inside_wall=f["inside_wall"],
                samples=int(f["samples"]),
                provenance=ast.literal_eval(str(f["provenance"][0])),
            )


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def matrix_path(ws, machine: str, name: str) -> Path:
    """Where a matrix lives: <cache>/<machine>/geometry/<name>.npz."""
    return ws.path("cache") / machine / "geometry" / f"{name}.npz"


def build_geometry_matrix(tokamak, cameras=None, resolution="medium", samples: int | None = None, progress=None) -> GeometryMatrix:
    """
    Build G for the tokamak's cameras (built, mode "Build", with bolometers).

    cameras    : camera names to include (default all)
    resolution : "low" | "medium" | "high" or {dR, dz, n_phi}
    samples    : rays per channel (default: the preset's)
    progress   : optional callable(text) for progress lines

    Each camera is observed with the other cameras removed from the world,
    as bolos_observe does, so their housings do not block its view.
    """
    from typing import Any, cast

    from raysect.core import translate
    from cherab.tools.raytransfer import RayTransferCylinder, RayTransferPipeline0D

    say = progress or (lambda text: None)
    if tokamak.wall is None or tokamak.world is None or not getattr(tokamak, "bolometers", None):
        raise RuntimeError("Build the tokamak (mode='Build', loadBolometers=True) before building a geometry matrix")
    grid = GridSpec.from_wall(tokamak.wall, resolution)
    if samples is None:
        samples = DEFAULT_SAMPLES[resolution] if isinstance(resolution, str) else DEFAULT_SAMPLES["medium"]
    selected = [b for b in tokamak.bolometers if cameras is None or b.name in cameras]
    if not selected:
        raise ValueError(f"no cameras to build for: {cameras} (the machine has {[b.name for b in tokamak.bolometers]})")
    tokamak.calc_etendues()

    # --- The voxel grid, in place of the emission volume, labels every voxel with its own spectral bin
    emission_surface = tokamak.cherab_objects.get("Emission Surface")
    if emission_surface is not None:
        emission_surface.parent = None
    rtc = RayTransferCylinder(
        radius_outer=float(grid.r_edges[-1]),
        height=float(grid.z_edges[-1] - grid.z_edges[0]),
        n_radius=grid.n_r,
        n_height=grid.n_z,
        radius_inner=cast(Any, float(grid.r_edges[0])),  # the stub says int; a float is what it takes
        n_polar=grid.n_phi,
        parent=tokamak.world,
        transform=translate(0, 0, float(grid.z_edges[0])),
    )
    voxel_map = np.asarray(rtc.voxel_map)  # (n_r, n_phi, n_z) -> spectral bin
    if voxel_map.shape != (grid.n_r, grid.n_phi, grid.n_z):
        raise RuntimeError(f"unexpected voxel map shape {voxel_map.shape}")
    bin_of_voxel = voxel_map.ravel()  # voxel j (C order) -> bin

    RR, ZZ = np.meshgrid(grid.r_centers, grid.z_centers, indexing="ij")
    inside_rz = tokamak._inside_tokamak(np.column_stack([RR.ravel(), ZZ.ravel()])).reshape(grid.n_r, grid.n_z)
    inside_wall = np.repeat(inside_rz[:, None, :], grid.n_phi, axis=1).ravel()

    say(f"geometry matrix for {tokamak.info['tokamakName']}: {grid.describe()}, {samples:,} rays per channel")
    rows, channels, cam_of_row, etendues = [], [], [], []
    # The foils' own settings, restored afterwards (the voxel bins would make later observes crawl)
    saved = {foil: (foil.units, foil.spectral_bins, foil.min_wavelength, foil.max_wavelength, foil.pixel_samples, foil.quiet)
             for b in selected for foil in b.bolometer_camera}
    t0 = time.perf_counter()
    try:
        for bolo in tokamak.bolometers:
            bolo._change_parent(value=None)
        for ci, bolo in enumerate(selected):
            bolo._change_parent(value=tokamak.world)
            tags = bolo.info["CHANNEL_TAGS"]
            for fi, foil in enumerate(bolo.bolometer_camera):
                foil.units = "Power"  # first: setting units recreates the foil's pipeline
                foil.pipelines = [RayTransferPipeline0D(kind="power")]
                foil.spectral_bins = rtc.bins
                foil.min_wavelength, foil.max_wavelength = 500.0, 500.0 + rtc.bins  # bins are voxel labels
                foil.pixel_samples = samples
                foil.quiet = True
                foil.observe()
                row_bins = np.asarray(foil.pipelines[0].matrix, dtype=float)  # per spectral bin [m^3 sr]
                row = row_bins[bin_of_voxel] / max(float(bolo.etendues[fi]), 1e-300)  # brightness per W/m^3
                rows.append(sp.csr_matrix(row))
                channels.append(str(tags[fi]))
                cam_of_row.append(ci)
                etendues.append(float(bolo.etendues[fi]))
                say(f"  {bolo.name} {tags[fi]}: {int((row > 0).sum())} voxels seen ({time.perf_counter() - t0:.0f} s)")
            bolo._change_parent(value=None)
    finally:
        for foil, (units, bins, wmin, wmax, px, quiet) in saved.items():
            foil.units = units  # recreates the foil's normal pipeline
            foil.spectral_bins, foil.min_wavelength, foil.max_wavelength = bins, wmin, wmax
            foil.pixel_samples, foil.quiet = px, quiet
        for bolo in tokamak.bolometers:
            bolo._change_parent(value=tokamak.world)
        rtc.parent = None
        if emission_surface is not None:
            emission_surface.parent = tokamak.world

    return GeometryMatrix(
        machine=str(tokamak.info["tokamakName"]),
        grid=grid,
        rows=sp.csr_matrix(sp.vstack(rows)),
        cameras=[b.name for b in selected],
        channels=channels,
        camera_of_row=np.asarray(cam_of_row),
        etendues=np.asarray(etendues),
        inside_wall=inside_wall,
        samples=int(samples),
        provenance={"built_in": round(time.perf_counter() - t0, 1), "resolution": resolution if isinstance(resolution, str) else "custom"},
    )
