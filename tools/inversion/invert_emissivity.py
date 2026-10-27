#!/usr/bin/env python
"""
invert_emissivity.py -- a regularized emissivity inversion through Emis3D's
geometry matrix. A standalone tool, outside the emis3d package: it shows the
emission shape the data want without assuming one, to guide the choice of
parametric models.

Modes
  axisymmetric   One (R, z) emissivity map, the same at every toroidal angle.
                 Kernel: the geometry matrix summed over toroidal sectors.
  field-aligned  The emission source on the injector's poloidal plane, carried
                 along field lines in both directions with a toroidal decay
                 exp(-b dphi^2). Kernel: G M, where M maps each source pixel
                 along its field lines into the voxels. Cameras at different
                 toroidal angles then see the source in different poloidal
                 places, as helical radiation would appear.

Both solve
    min  sum_i ((d_i - (K x)_i) / sigma_i)^2 + lam * ||L x||^2,   x >= 0
with L the 2-D Laplacian on the pixel grid. lam is chosen so chi2/N is the
target (discrepancy principle: as smooth as the data's errors allow), unless
given.

Usage (from the workspace, with the emis3d environment active)
  python tools/inversion/invert_emissivity.py runs/JET/95709/95709_runConfig.yaml --time 50.9543
  python tools/inversion/invert_emissivity.py RUN --time T --mode field-aligned --decay 0.3
  options: --matrix medium  --equilibrium <g-file>  --transits 1  --substeps 2
           --lam 1e-3  --target-chi2 1.0  --psi-max 1.2  --out <folder>
The field-aligned mode's field-line mapping is cached beside the geometry
matrix (emis3d.fieldline_map), shared with the parametric helical model.

Outputs: <out>/inversion_<mode>_t<time>.png (maps, fits, lambda scan) and
.npz (solution, grid, kernel diagnostics). The run config's data, errors and
bad channels are used as for a fit.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml

from emis3d.fieldline_map import pixel_grid  # noqa: F401  (re-exported for the tests and callers)

logger = logging.getLogger("invert_emissivity")


# ---------------------------------------------------------------------------
# Data and matrix
# ---------------------------------------------------------------------------


def load_problem(run_config, evalTime: float, matrix_name: str):
    """
    The run's data at evalTime (channels, values, errors) and the geometry
    matrix, through emis3d. The run config is read as a parametric run (no
    radDist library is loaded) from a temporary copy beside it.
    """
    from emis3d.emis3D import Emis3D
    from emis3d.geometry import cached_matrix, matrix_path

    run_config = Path(run_config).resolve()
    cfg = yaml.safe_load(run_config.read_text())
    cfg["solver"] = "parametric"
    equilibrium = (cfg.get("parametric") or {}).get("equilibrium")
    cfg["parametric"] = {"models": ["ring"], "matrices": [matrix_name], **({"equilibrium": equilibrium} if equilibrium else {})}
    tmp = run_config.with_name(f".invert_{run_config.stem}.yaml")
    tmp.write_text(yaml.safe_dump(cfg))
    try:
        e = Emis3D(runConfigName=str(tmp))
    finally:
        tmp.unlink(missing_ok=True)

    path = matrix_path(e.ws, e.info["tokamakName"], matrix_name)
    if not path.is_file():
        raise FileNotFoundError(f"no geometry matrix {matrix_name!r} at {path}; build it with emis3d build-matrix {e.info['tokamakName']} --resolution {matrix_name}")
    G = cached_matrix(path)
    e._prune_channel_order(set(G.channels), "the geometry matrix")
    e._check_time_in_data(evalTime)
    e._prepare_data_for_fit(evalTime=evalTime)

    tags, d, sig, cams = [], [], [], []
    for ii, bolo in enumerate(e.channel_order["bolometer_order"]):
        vals = e.fitData[evalTime]["boloData"][bolo]
        errs = e.fitData[evalTime]["boloData_error"][bolo]
        for tag, v, s in zip(e.channel_order["channel_list"][ii], vals, errs, strict=True):
            if np.isfinite(s) and s > 0:  # bad channels have infinite error
                tags.append(str(tag))
                d.append(float(v))
                sig.append(float(s))
                cams.append(bolo)
    rows = np.array([G.channels.index(t) for t in tags])
    return e, G, rows, np.array(d), np.array(sig), tags, cams


def laplacian(ir, iz, index) -> sp.csr_matrix:
    """The graph Laplacian of the pixels (4-neighbours): (L x)_p = sum over neighbours (x_p - x_q)."""
    rows, cols, vals = [], [], []
    n_r, n_z = index.shape
    for p, (r, z) in enumerate(zip(ir, iz, strict=True)):
        nb = [index[r + dr, z + dz] for dr, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= r + dr < n_r and 0 <= z + dz < n_z and index[r + dr, z + dz] >= 0]
        rows.append(p)
        cols.append(p)
        vals.append(float(len(nb)))
        for q in nb:
            rows.append(p)
            cols.append(q)
            vals.append(-1.0)
    return sp.csr_matrix((vals, (rows, cols)), shape=(ir.size, ir.size))


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def kernel_axisymmetric(G, rows, ir, iz) -> sp.csr_matrix:
    """K (channels x pixels): the matrix summed over toroidal sectors for each (R, z) pixel."""
    g = G.grid
    vox, pix = [], []
    for p, (r, z) in enumerate(zip(ir, iz, strict=True)):
        k = np.arange(g.n_phi)
        vox.append(G.eps_index(r, k, z))
        pix.append(np.full(g.n_phi, p))
    S = sp.csr_matrix((np.ones(sum(v.size for v in vox)), (np.concatenate(vox), np.concatenate(pix))), shape=(g.n_voxels, ir.size))
    return sp.csr_matrix(G.rows[rows] @ S)


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------


class Problem:
    """
    The weighted least-squares problem, prepared once for every lambda:
    f(x) = ||(d - K x) / sigma||^2 + lam * scale * ||L x||^2,  x >= 0.
    scale makes lam relative (K's and L's sizes differ by orders of magnitude).
    """

    def __init__(self, K, d, sig, L):
        self.Kw = np.asarray(sp.csr_matrix(K).multiply(1.0 / sig[:, None]).toarray(), dtype=float)  # few rows: dense
        self.b = d / sig
        self.L = sp.csr_matrix(L)
        self.LtL = sp.csr_matrix(self.L.T @ self.L)
        self.scale = float(np.sum(self.Kw**2) / max(self.L.multiply(self.L).sum(), 1e-300))
        self.k_norm2 = float(np.linalg.norm(self.Kw, 2) ** 2)  # largest singular value squared
        self.l_norm2 = float(sp.linalg.norm(self.LtL, 1))  # an upper bound of ||L||_2^2

    def objective(self, x: np.ndarray, mu: float) -> tuple[float, np.ndarray]:
        r = self.Kw @ x - self.b
        Lx = self.L @ x
        return float(r @ r + mu * (Lx @ Lx)), 2.0 * (self.Kw.T @ r) + 2.0 * mu * (self.LtL @ x)

    def optimality(self, x: np.ndarray, mu: float) -> float:
        """
        The projected gradient's size at x relative to the gradient's size at
        the start (x = 0): 0 at the constrained optimum, 1 for no progress.
        """
        _, g = self.objective(x, mu)
        _, g0 = self.objective(np.zeros_like(x), mu)
        pg = np.where(x > 0, g, np.minimum(g, 0.0))
        return float(np.abs(pg).max() / max(np.abs(g0).max(), 1e-300))

    def solve(self, lam: float, x0=None, max_iter: int = 20000) -> tuple[np.ndarray, int, float]:
        """L-BFGS-B with x >= 0; returns (x, iterations, optimality)."""
        from scipy.optimize import minimize

        mu = lam * self.scale
        x_start = np.zeros(self.Kw.shape[1]) if x0 is None else np.maximum(np.asarray(x0, dtype=float), 0.0)
        res = minimize(lambda v: self.objective(v, mu), x_start, jac=True, method="L-BFGS-B",
                       bounds=[(0.0, None)] * x_start.size, options={"maxiter": max_iter, "maxfun": 2 * max_iter, "ftol": 1e-15, "gtol": 1e-14})
        x = np.asarray(res.x, dtype=float)
        return x, int(res.nit), self.optimality(x, mu)


def chi2_per_point(K, x, d, sig) -> float:
    return float(np.mean(((d - K @ x) / sig) ** 2))


def scan(K, d, sig, L, lams, target: float, progress=None):
    """
    Solve from the strongest to the weakest lambda (each from the previous
    solution); pick the largest lambda whose chi2/N <= target, else the
    best-fitting.
    """
    say = progress or (lambda text: None)
    prob = Problem(K, d, sig, L)
    results, x = [], None
    for lam in sorted(lams, reverse=True):
        t0 = time.perf_counter()
        x, its, opt = prob.solve(lam, x0=x)
        c = chi2_per_point(K, x, d, sig)
        rough = float(np.linalg.norm(L @ x))
        results.append((lam, x.copy(), c, rough))
        say(f"  lam {lam:9.2e}: chi2/N {c:8.3f}, roughness {rough:.3g} ({its} iterations, optimality {opt:.1e}, {time.perf_counter() - t0:.1f} s)")
    results.sort(key=lambda r: r[0])
    ok = [r for r in results if r[2] <= target]
    chosen = max(ok, key=lambda r: r[0]) if ok else min(results, key=lambda r: r[2])
    if not ok:
        say(f"  no regularization reaches chi2/N {target:g}: the best is {chosen[2]:.2f} at lam {chosen[0]:.1e}")
    return chosen, results


def per_camera_chi2(K, x, d, sig, cams) -> dict:
    model = np.asarray(K @ x).ravel()
    out = {}
    for cam in dict.fromkeys(cams):
        sel = np.array([c == cam for c in cams])
        out[cam] = float(np.mean(((d[sel] - model[sel]) / sig[sel]) ** 2))
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def camera_angles(G, tok, tags, cams) -> dict:
    """
    Toroidal angle [rad] of each camera as the run config names it. The run
    config's names (BOLOMETERS keys) need not match the machine's camera
    objects' names, so the angle is found through the channels: each
    channel is a row of the geometry matrix, which records the camera object
    it came from (the same objects the angles are read from).
    """
    by_object = {b.name: float(np.deg2rad(float(b.info["CAMERA_POSITION_R_Z_PHI"][2]))) % (2 * np.pi) for b in tok.bolometers}
    row_of = {t: i for i, t in enumerate(G.channels)}
    out = {}
    for cam in dict.fromkeys(cams):
        objects = [G.cameras[G.camera_of_row[row_of[t]]] for t, c in zip(tags, cams, strict=True) if c == cam and t in row_of]
        angles = [by_object[o] for o in objects if o in by_object]
        if angles:
            out[cam] = float(np.angle(np.mean(np.exp(1j * np.array(angles))))) % (2 * np.pi)  # all its channels' cameras
        elif cam in by_object:
            out[cam] = by_object[cam]
    return out


def as_map(values, ir, iz, shape) -> np.ndarray:
    out = np.full(shape, np.nan)
    out[ir, iz] = values
    return out


def plot(path, mode, G, ir, iz, x, eps_vox, K, d, sig, tags, cams, chosen, results, phi0, camera_phi, psi_norm, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = G.grid
    shape = (g.n_r, g.n_z)
    RR, ZZ = np.meshgrid(g.r_centers, g.z_centers, indexing="ij")
    psin = psi_norm(RR.ravel(), ZZ.ravel()).reshape(shape) if psi_norm is not None else None
    cam_names = list(dict.fromkeys(cams))

    maps = [("reconstruction" if mode == "axisymmetric" else f"source on the plane at {np.rad2deg(phi0):.0f} deg", as_map(x, ir, iz, shape))]
    if mode == "field-aligned":
        for cam in cam_names:
            if cam not in camera_phi:
                logger.warning("no toroidal angle found for camera %s: its emissivity map is left out", cam)
                continue
            k = int(np.argmin(np.abs((g.phi_centers - camera_phi[cam] + np.pi) % (2 * np.pi) - np.pi)))
            plane = eps_vox.reshape(g.n_r, g.n_phi, g.n_z)[:, k, :]
            maps.append((f"emissivity at {cam} ({np.rad2deg(camera_phi[cam]):.0f} deg)", np.where(plane > 0, plane, np.nan)))

    n_maps = len(maps)
    fig = plt.figure(figsize=(4.2 * max(n_maps, len(cam_names) + 1), 10))
    gs = fig.add_gridspec(2, max(n_maps, len(cam_names) + 1))
    for j, (label, m) in enumerate(maps):
        ax = fig.add_subplot(gs[0, j])
        finite = m[np.isfinite(m)]
        vmax = float(np.percentile(finite[finite > 0], 99.5)) if np.any(finite > 0) else None  # a few hot pixels must not hide the map
        pc = ax.pcolormesh(g.r_centers, g.z_centers, m.T, shading="nearest", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.contour(g.r_centers, g.z_centers, (G.inside_wall.reshape(g.n_r, g.n_phi, g.n_z)[:, 0, :]).T.astype(float), levels=[0.5], colors="w", linewidths=0.8)
        if psin is not None:
            ax.contour(g.r_centers, g.z_centers, psin.T, levels=[1.0], colors="r", linewidths=0.8, linestyles="--")
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("z [m]")
        fig.colorbar(pc, ax=ax, shrink=0.8, label="emissivity [W/m3]" + (f" (scale to 99.5th percentile, max {np.nanmax(m):.2g})" if vmax else ""))

    model = np.asarray(K @ x).ravel()
    for j, cam in enumerate(cam_names):
        sel = np.array([c == cam for c in cams])
        ax = fig.add_subplot(gs[1, j])
        n = np.arange(sel.sum())
        ax.errorbar(n, d[sel], yerr=sig[sel], fmt="ks", ms=3, label="data")
        ax.plot(n, model[sel], "-", color="purple", label="inversion")
        c = float(np.mean(((d[sel] - model[sel]) / sig[sel]) ** 2))
        ax.set_title(f"{cam}  chi2/N = {c:.2f}  ({sel.sum()} channels)", fontsize=9)
        ax.set_xlabel("channel (fit order)")
        ax.set_ylabel("brightness [W/m2]")
        ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, len(cam_names)])
    lams = [r[0] for r in results]
    ax.loglog(lams, [r[2] for r in results], "o-", label="chi2/N")
    ax.axvline(chosen[0], color="r", ls="--", label=f"chosen {chosen[0]:.1e}")
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.set_xlabel("lambda (relative)")
    ax.set_ylabel("chi2/N")
    ax.legend(fontsize=7)
    ax.set_title("regularization scan", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_config")
    ap.add_argument("--time", type=float, required=True, help="evaluation time")
    ap.add_argument("--matrix", default="medium", help="geometry matrix name in the cache (default medium)")
    ap.add_argument("--mode", choices=["axisymmetric", "field-aligned"], default="axisymmetric")
    ap.add_argument("--equilibrium", help="g-file (default: the run config's parametric.equilibrium); needed for field-aligned")
    ap.add_argument("--decay", type=float, default=0.0, help="field-aligned: toroidal decay b in exp(-b dphi^2), dphi in rad (default 0)")
    ap.add_argument("--transits", type=int, default=1, help="field-aligned: toroidal turns followed in each direction (default 1)")
    ap.add_argument("--substeps", type=int, default=2, help="field-aligned: field lines per pixel edge (default 2)")
    ap.add_argument("--psi-max", type=float, default=None, help="only pixels with psi_N <= this (default: all inside the wall)")
    ap.add_argument("--source-plane", type=float, default=None,
                    help="field-aligned: toroidal angle [deg] of the source plane (default: the injector); the injector's "
                         "mapping is rotated there, no retracing")
    ap.add_argument("--mirror-helicity", action="store_true",
                    help="field-aligned: wind the field lines the opposite way (a check of the sign conventions: with "
                         "the right ones the mirrored case needs a rougher map, i.e. weaker regularization, to fit)")
    ap.add_argument("--lam", type=float, default=None, help="regularization (relative); default: scanned, chi2/N = target")
    ap.add_argument("--target-chi2", type=float, default=1.0, help="chi2/N the scan aims for (default 1)")
    ap.add_argument("--out", help="output folder (default: the run config's folder)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    say = logger.info

    e, G, rows, d, sig, tags, cams = load_problem(args.run_config, args.time, args.matrix)
    say(f"{e.info['tokamakName']} {e.info['shot']} t={args.time}: {d.size} channels, matrix {G.grid.describe()}")

    from emis3d.tokamak import Tokamak

    eq_name = args.equilibrium or (((e.config.parametric or {}) if e.config else {}).get("equilibrium"))
    # One tokamak gives the equilibrium (if any) and the cameras' toroidal angles
    tok = Tokamak(tokamakName=e.info["tokamakName"], mode="Build", reflections=False, eqFileName=eq_name, loadBolometers=True, workspace=e.ws)
    camera_phi = camera_angles(G, tok, tags, cams)
    psi_norm, gfile = None, None
    if eq_name:
        gfile = tok.gfile
        if gfile is not None:
            from scipy.interpolate import RegularGridInterpolator

            Rg = np.linspace(gfile.rleft, gfile.rleft + gfile.rdim, gfile.nx)
            Zg = np.linspace(gfile.zmid - gfile.zdim / 2, gfile.zmid + gfile.zdim / 2, gfile.ny)
            interp = RegularGridInterpolator((Rg, Zg), (np.asarray(gfile.psi, float) - gfile.simagx) / (gfile.sibdry - gfile.simagx),
                                             bounds_error=False, fill_value=1e6)
            psi_norm = lambda R, Z: interp(np.column_stack([R, Z]))  # noqa: E731
    if args.mode == "field-aligned" and gfile is None:
        raise SystemExit("field-aligned mode needs an equilibrium: --equilibrium <g-file> (or parametric.equilibrium in the run config)")

    ir, iz, index = pixel_grid(G, psi_norm, args.psi_max)
    L = laplacian(ir, iz, index)
    phi0 = float(np.deg2rad(e.info["injectionLocation"])) % (2.0 * np.pi)
    shift = 0.0
    if args.mode == "axisymmetric":
        K = kernel_axisymmetric(G, rows, ir, iz)
        M = None
    else:
        # the cached field-line mapping of this matrix (emis3d.fieldline_map; also the helical model's)
        from emis3d.fieldline_map import get_or_build

        fmap, mpath = get_or_build(e.ws, G, e.info["tokamakName"], args.matrix, eq_name, float(e.info["injectionLocation"]),
                                   transits=args.transits, substeps=args.substeps, psi_max=args.psi_max, progress=say,
                                   mirror_helicity=args.mirror_helicity)
        say(f"field-line mapping: {mpath} (decay {args.decay})")
        ir, iz = fmap.ir, fmap.iz  # the mapping's own source pixels
        _, _, index = pixel_grid(G, psi_norm, args.psi_max)
        L = laplacian(ir, iz, index)
        shift = (np.deg2rad(args.source_plane) - phi0) if args.source_plane is not None else 0.0
        if shift:
            say(f"source plane rotated to {args.source_plane:g} deg ({np.rad2deg(shift):+.1f} deg from the injector)")
        M = fmap.matrix(args.decay, shift=shift)
        K = sp.csr_matrix(G.rows[rows] @ M)

    lams = [args.lam] if args.lam is not None else list(np.logspace(-8, 4, 25))
    chosen, results = scan(K, d, sig, L, lams, args.target_chi2, progress=say)
    lam, x, c, rough = chosen

    g = G.grid
    if M is None:
        eps_vox = np.zeros(g.n_voxels)
        for p, (r, z) in enumerate(zip(ir, iz, strict=True)):
            eps_vox[G.eps_index(r, np.arange(g.n_phi), z)] = x[p]
    else:
        eps_vox = np.asarray(M @ x).ravel()
    volume = (g.r_centers[:, None, None] * np.diff(g.r_edges).mean() * np.diff(g.z_edges).mean() * (2 * np.pi / g.n_phi)) * np.ones((1, g.n_phi, g.n_z))
    P_sector = (eps_vox.reshape(g.n_r, g.n_phi, g.n_z) * volume).sum(axis=(0, 2))
    total = float(P_sector.sum())
    tpf = float(P_sector.max() / P_sector.mean()) if P_sector.mean() > 0 else float("nan")
    say(f"chosen lambda {lam:.2e}: chi2/N {c:.2f}, roughness {rough:.3g}; total radiated power {total:.3g} W; TPF {tpf:.2f}"
        + (" (axisymmetric by construction)" if M is None else ""))
    by_cam = per_camera_chi2(K, x, d, sig, cams)
    say("chi2/N per camera: " + ", ".join(f"{cam} {v:.2f}" for cam, v in by_cam.items()))
    if c > args.target_chi2:
        say(f"NOTE: no {args.mode} emissivity fits these channels within their errors, however weak the regularization. "
            "Either the emission is not " + ("axisymmetric (toroidally localized, e.g. near the injector): try --mode field-aligned,"
                                             if args.mode == "axisymmetric" else "field-aligned with this decay: try other --decay values,")
            + " or a camera's calibration or geometry disagrees with the others (see the chi2/N per camera).")

    out = Path(args.out) if args.out else Path(args.run_config).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"inversion_{args.mode}_t{args.time:g}" if args.mode == "axisymmetric" else \
        out / f"inversion_{args.mode}_t{args.time:g}_decay{args.decay:g}{f'_plane{args.source_plane:g}' if args.source_plane is not None else ''}{'_mirrored' if args.mirror_helicity else ''}"
    title = (f"{e.info['tokamakName']} {e.info['shot']}  t = {args.time:g}  {args.mode} inversion"
             + (f" (decay {args.decay:g}, {args.transits} transit(s)" + (f", source plane {args.source_plane:g} deg" if args.source_plane is not None else "")
                + f"{', HELICITY MIRRORED' if args.mirror_helicity else ''})" if M is not None else "")
             + f"\nlambda {lam:.1e}   chi2/N {c:.2f}   roughness {rough:.3g}   total power {total:.3g} W   TPF {tpf:.2f}")
    plane = phi0 + shift if M is not None else phi0
    plot(stem.with_suffix(".png"), args.mode, G, ir, iz, x, eps_vox, K, d, sig, tags, cams, chosen, results, plane, camera_phi, psi_norm, title)
    np.savez_compressed(stem.with_suffix(".npz"), x=x, ir=ir, iz=iz, r_centers=g.r_centers, z_centers=g.z_centers, phi_centers=g.phi_centers,
                        P_sector=P_sector, lam=lam, chi2_per_point=c, tags=np.array(tags), data=d, sigma=sig, model=np.asarray(K @ x).ravel(),
                        mode=args.mode, decay=args.decay)
    say(f"saved {stem.with_suffix('.png')} and .npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
