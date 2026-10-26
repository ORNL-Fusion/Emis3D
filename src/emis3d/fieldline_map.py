# fieldline_map.py
"""
A field-line mapping on a geometry matrix's voxel grid: emission from a
poloidal plane (the injector's) carried along traced field lines in both
directions, with a toroidal decay. This is the helical picture of injected
radiation (a source near the injector spreading along the field), built
once per machine, equilibrium and matrix, and cached: afterwards any source
on the plane becomes a voxel emissivity in milliseconds, and the decay can
vary without retracing.

    eps(voxel) = sum over passes  exp(-decay * dphi_pass(voxel)^2) * (M_pass . source)

where a pass is a direction (+/-) and a transit, M_pass carries each
source pixel to the voxels its field line lands in at every toroidal
sector (a voxel takes the mean of the sources whose lines land in it), and
dphi is the toroidal angle travelled to that voxel's sector on that pass.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def pixel_grid(G, psi_norm=None, psi_max=None):
    """The (R, z) cells of the matrix's grid inside the wall (and, if given, psi_N <= psi_max): (ir, iz, index[ir, iz])."""
    g = G.grid
    inside = G.inside_wall.reshape(g.n_r, g.n_phi, g.n_z)[:, 0, :].copy()
    if psi_norm is not None and psi_max is not None:
        RR, ZZ = np.meshgrid(g.r_centers, g.z_centers, indexing="ij")
        inside &= np.asarray(psi_norm(RR.ravel(), ZZ.ravel())).reshape(RR.shape) <= psi_max
    ir, iz = np.nonzero(inside)
    index = -np.ones((g.n_r, g.n_z), dtype=int)
    index[ir, iz] = np.arange(ir.size)
    return ir, iz, index


def map_path(ws, machine: str, matrix_name: str, equilibrium: str, phi0_deg: float, transits: int) -> Path:
    """Where a mapping lives: <cache>/<machine>/geometry/<matrix>_map_<equilibrium>_<phi0>deg_<transits>t.npz."""
    eq = Path(str(equilibrium)).stem.replace(" ", "_")
    return ws.path("cache") / machine / "geometry" / f"{matrix_name}_map_{eq}_{phi0_deg:g}deg_{int(transits)}t.npz"


@dataclass
class FieldLineMap:
    machine: str
    matrix_name: str
    equilibrium: str
    phi0: float  # the source plane's toroidal angle [rad]
    transits: int
    substeps: int
    n_voxels: int
    n_phi: int
    phi_centers: np.ndarray
    sector_of_voxel: np.ndarray
    ir: np.ndarray  # source pixels: grid cell indices
    iz: np.ndarray
    r_centers: np.ndarray
    z_centers: np.ndarray
    passes: list = field(default_factory=list)  # (direction, transit, M_pass csr (voxels x pixels))
    provenance: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def n_pixels(self) -> int:
        return int(self.ir.size)

    @property
    def pixel_R(self) -> np.ndarray:
        return self.r_centers[self.ir]

    @property
    def pixel_z(self) -> np.ndarray:
        return self.z_centers[self.iz]

    def travelled(self, direction: int, transit: int) -> np.ndarray:
        """Toroidal angle travelled [rad] to each voxel's sector on a pass."""
        two_pi = 2.0 * np.pi
        return (direction * (self.phi_centers[self.sector_of_voxel] - self.phi0)) % two_pi + transit * two_pi

    def apply(self, source: np.ndarray, decay: float) -> np.ndarray:
        """The voxel emissivity of a source on the plane (per pixel), with decay exp(-decay dphi^2)."""
        eps = np.zeros(self.n_voxels)
        for direction, transit, M in self.passes:
            w = np.exp(-float(decay) * self.travelled(direction, transit) ** 2)
            eps += w * np.asarray(M @ np.asarray(source, dtype=float)).ravel()
        return eps

    def matrix(self, decay: float) -> sp.csr_matrix:
        """The mapping as one sparse matrix (voxels x pixels) for a decay."""
        total = sp.csr_matrix((self.n_voxels, self.n_pixels))
        for direction, transit, M in self.passes:
            w = np.exp(-float(decay) * self.travelled(direction, transit) ** 2)
            total = total + sp.diags(w) @ M
        return sp.csr_matrix(total)

    def gaussian_source(self, R0: float, z0: float, sigma_R: float, sigma_z: float) -> np.ndarray:
        """A Gaussian source on the plane, per pixel (peak 1)."""
        return np.exp(-0.5 * (((self.pixel_R - R0) / sigma_R) ** 2 + ((self.pixel_z - z0) / sigma_z) ** 2))

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, G, gfile, phi0: float, matrix_name: str, equilibrium: str, transits: int = 1, substeps: int = 2,
              psi_norm=None, psi_max: float | None = None, n_tor: int | None = None, progress=None) -> "FieldLineMap":
        """
        Trace every source pixel's field line (substeps x substeps lines per
        pixel, both directions, `transits` turns) and record, per pass, the
        voxels each line lands in at every sector.
        """
        from emis3d.fieldline_tracer import Fieldline_Tracer

        say = progress or (lambda text: None)
        g = G.grid
        two_pi = 2.0 * np.pi
        n_tor = n_tor or max(300, 8 * g.n_phi)
        ir, iz, _ = pixel_grid(G, psi_norm, psi_max)
        dR, dz = np.diff(g.r_edges).mean(), np.diff(g.z_edges).mean()
        offs = (np.arange(substeps) + 0.5) / substeps - 0.5
        inside_rz = G.inside_wall.reshape(g.n_r, g.n_phi, g.n_z)[:, 0, :]
        sector = G.sector_of_voxel()

        hits: dict = {}
        t0 = time.perf_counter()
        say(f"tracing field lines from {ir.size} pixels on the plane {np.rad2deg(phi0):.1f} deg "
            f"({substeps * substeps} per pixel, both directions, {transits} transit(s))")
        for p, (r, z) in enumerate(zip(ir, iz, strict=True)):
            for a in offs:
                for b in offs:
                    R0, Z0 = g.r_centers[r] + a * dR, g.z_centers[z] + b * dz
                    for direction in (1, -1):
                        tr = Fieldline_Tracer(StartR=R0, StartZ=Z0, StartPhi=phi0, gfile=gfile, NumTor=n_tor)
                        tr.trace(NumTransits=float(direction * transits))
                        phi_line = np.asarray(tr.data["phi"], dtype=float)
                        travelled = direction * (phi_line - phi0)
                        Rl, Zl = np.asarray(tr.data["R"], dtype=float), np.asarray(tr.data["z"], dtype=float)
                        for m in range(transits):
                            delta = (direction * (g.phi_centers - phi0)) % two_pi + m * two_pi
                            Rk = np.interp(delta, travelled, Rl, left=np.nan, right=np.nan)
                            Zk = np.interp(delta, travelled, Zl, left=np.nan, right=np.nan)
                            i_r = np.searchsorted(g.r_edges, Rk) - 1
                            i_z = np.searchsorted(g.z_edges, Zk) - 1
                            ok = np.isfinite(Rk) & (i_r >= 0) & (i_r < g.n_r) & (i_z >= 0) & (i_z < g.n_z)
                            ok[ok] &= inside_rz[i_r[ok], i_z[ok]]
                            k = np.flatnonzero(ok)
                            lst = hits.setdefault((direction, m), ([], []))
                            lst[0].append(G.eps_index(i_r[k], k, i_z[k]))
                            lst[1].append(np.full(k.size, p))
            if p and p % 250 == 0:
                say(f"  traced {p}/{ir.size} pixels ({time.perf_counter() - t0:.0f} s)")

        passes = []
        for (direction, m), (vox_l, pix_l) in sorted(hits.items()):
            vox, pix = np.concatenate(vox_l), np.concatenate(pix_l)
            H = sp.csr_matrix((np.ones(vox.size), (vox, pix)), shape=(g.n_voxels, ir.size))
            count = np.asarray(H.sum(axis=1)).ravel()
            M = sp.csr_matrix(sp.diags(np.where(count > 0, 1.0 / np.maximum(count, 1.0), 0.0)) @ H)
            passes.append((int(direction), int(m), M))
        say(f"mapping built in {time.perf_counter() - t0:.0f} s: {len(passes)} passes, {sum(M.nnz for _, _, M in passes):,} nonzeros")
        return cls(machine=G.machine, matrix_name=matrix_name, equilibrium=str(equilibrium), phi0=float(phi0), transits=int(transits),
                   substeps=int(substeps), n_voxels=g.n_voxels, n_phi=g.n_phi, phi_centers=g.phi_centers, sector_of_voxel=sector,
                   ir=ir, iz=iz, r_centers=g.r_centers, z_centers=g.z_centers, passes=passes,
                   provenance={"built_in": round(time.perf_counter() - t0, 1), "n_tor": n_tor, "psi_max": psi_max})

    # ------------------------------------------------------------------
    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "machine": self.machine, "matrix_name": self.matrix_name, "equilibrium": self.equilibrium, "phi0": self.phi0,
            "transits": self.transits, "substeps": self.substeps, "n_voxels": self.n_voxels, "n_phi": self.n_phi,
            "phi_centers": self.phi_centers, "sector_of_voxel": self.sector_of_voxel, "ir": self.ir, "iz": self.iz,
            "r_centers": self.r_centers, "z_centers": self.z_centers, "n_passes": len(self.passes),
            "provenance": np.array([str(self.provenance)]),
        }
        for k, (direction, m, M) in enumerate(self.passes):
            arrays[f"pass{k}_dir"] = direction
            arrays[f"pass{k}_transit"] = m
            arrays[f"pass{k}_data"] = M.data
            arrays[f"pass{k}_indices"] = M.indices
            arrays[f"pass{k}_indptr"] = M.indptr
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path) -> "FieldLineMap":
        import ast

        with np.load(path, allow_pickle=False) as f:
            n_vox, n_pix = int(f["n_voxels"]), int(f["ir"].size)
            passes = []
            for k in range(int(f["n_passes"])):
                M = sp.csr_matrix((f[f"pass{k}_data"], f[f"pass{k}_indices"], f[f"pass{k}_indptr"]), shape=(n_vox, n_pix))
                passes.append((int(f[f"pass{k}_dir"]), int(f[f"pass{k}_transit"]), M))
            return cls(machine=str(f["machine"]), matrix_name=str(f["matrix_name"]), equilibrium=str(f["equilibrium"]), phi0=float(f["phi0"]),
                       transits=int(f["transits"]), substeps=int(f["substeps"]), n_voxels=n_vox, n_phi=int(f["n_phi"]),
                       phi_centers=f["phi_centers"], sector_of_voxel=f["sector_of_voxel"], ir=f["ir"], iz=f["iz"],
                       r_centers=f["r_centers"], z_centers=f["z_centers"], passes=passes,
                       provenance=ast.literal_eval(str(f["provenance"][0])))


_LOADED: dict = {}


def cached_map(path) -> FieldLineMap:
    key = str(Path(path).resolve())
    if key not in _LOADED:
        _LOADED[key] = FieldLineMap.load(key)
    return _LOADED[key]


def get_or_build(ws, G, machine: str, matrix_name: str, equilibrium: str, phi0_deg: float, transits: int = 1, substeps: int = 2,
                 psi_max: float | None = None, progress=None) -> tuple[FieldLineMap, Path]:
    """The cached mapping for these settings, built (and saved) if it does not exist yet."""
    path = map_path(ws, machine, matrix_name, equilibrium, phi0_deg, transits)
    if path.is_file():
        return cached_map(path), path
    from emis3d.tokamak import Tokamak

    tok = Tokamak(tokamakName=machine, mode="Analysis", eqFileName=equilibrium, workspace=ws)
    if tok.gfile is None:
        raise RuntimeError(f"equilibrium {equilibrium} did not load for {machine}")
    psi_norm = None
    if psi_max is not None:
        from scipy.interpolate import RegularGridInterpolator

        gf = tok.gfile
        Rg = np.linspace(gf.rleft, gf.rleft + gf.rdim, gf.nx)
        Zg = np.linspace(gf.zmid - gf.zdim / 2, gf.zmid + gf.zdim / 2, gf.ny)
        interp = RegularGridInterpolator((Rg, Zg), (np.asarray(gf.psi, float) - gf.simagx) / (gf.sibdry - gf.simagx), bounds_error=False, fill_value=1e6)
        psi_norm = lambda R, Z: interp(np.column_stack([R, Z]))  # noqa: E731
    fmap = FieldLineMap.build(G, tok.gfile, float(np.deg2rad(phi0_deg)) % (2.0 * np.pi), matrix_name, equilibrium, transits, substeps,
                              psi_norm, psi_max, progress=progress)
    fmap.provenance.update(ws.provenance())
    fmap.save(path)
    _LOADED[str(path.resolve())] = fmap
    return fmap, path
