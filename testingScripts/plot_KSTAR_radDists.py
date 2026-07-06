#!/usr/bin/env python3
"""Quick-look toroidal slice plots for recently generated KSTAR radDists."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main.Globals import EMIS3D_INPUTS_DIRECTORY
from main.radDist import ElongatedRing, Helical, HelicalRing


DEFAULT_ROOT = EMIS3D_INPUTS_DIRECTORY / "KSTAR" / "radDists"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one KSTAR helical radDist and one elongated-ring radDist at "
            "PAA_D, PAA_O, and injection toroidal locations."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root folder to search for radDist JSON files.",
    )
    parser.add_argument(
        "--helical-json",
        type=Path,
        default=None,
        help="Specific helical JSON to plot. Defaults to newest saved helical JSON.",
    )
    parser.add_argument(
        "--elongated-json",
        type=Path,
        default=None,
        help=(
            "Specific elongated-ring JSON to plot. Defaults to newest saved "
            "elongated-ring JSON."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT,
        help="Folder where the two PNG figures will be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive windows after saving.",
    )
    return parser.parse_args()


def read_info(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)["info"]


def newest_rad_dist_json(root: Path, dist_type: str) -> Path:
    matches = []
    for path in root.glob("**/*.json"):
        if not path.is_file():
            continue
        try:
            info = read_info(path)
        except (json.JSONDecodeError, KeyError):
            continue
        if str(info.get("distType", "")).lower() == dist_type.lower():
            matches.append(path)

    if not matches:
        raise FileNotFoundError(f"No {dist_type} radDist JSON files found under {root}")

    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]


def load_saved_rad_dist(path: Path):
    with path.open("r") as handle:
        saved = json.load(handle)

    info = saved["info"]
    dist_type = str(info["distType"]).lower()
    start_r = float(info["startR"])
    start_z = float(info["startZ"])

    if dist_type == "helical":
        rad_dist = Helical(startR=start_r, startZ=start_z, config=info)
    elif dist_type == "helicalring":
        rad_dist = HelicalRing(startR=start_r, startZ=start_z, config=info)
    elif dist_type == "elongatedring":
        rad_dist = ElongatedRing(startR=start_r, startZ=start_z, config=info)
    else:
        raise ValueError(f"Unsupported distType '{info['distType']}' in {path}")

    rad_dist.data = saved.get("data", {})
    return rad_dist


def bolometer_phi_by_prefix(rad_dist, group_prefix: str) -> float:
    phis = [
        float(bolo.info["CAMERA_POSITION_R_Z_PHI"][2])
        for bolo in rad_dist.tokamak.bolometers
        if bolo.info["GROUP_NAME"].startswith(group_prefix)
    ]
    if not phis:
        raise ValueError(f"No bolometer groups found with prefix {group_prefix}")
    return float(np.mean(phis))


def plot_slice(fig, ax, rad_dist, phi_deg: float, title: str) -> None:
    rad_dist.tokamak._plot_first_wall(ax)
    contour = rad_dist.plotCrossSection(phi=np.deg2rad(phi_deg), ax=ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]", fontsize=10)
    ax.set_ylabel("z [m]", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_title(f"{title}\nphi={phi_deg:.1f} deg", fontsize=10)
    colorbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label("emissivity", fontsize=9)


def plot_rad_dist(path: Path, output_path: Path, show: bool) -> None:
    print(f"Plotting {path}")
    rad_dist = load_saved_rad_dist(path)

    injection_phi = float(rad_dist.info["injectionLocation"])
    locations = [
        ("PAA_D", bolometer_phi_by_prefix(rad_dist, "PAA_D")),
        ("PAA_O", bolometer_phi_by_prefix(rad_dist, "PAA_O")),
        ("Injection", injection_phi),
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12, 4.4),
        constrained_layout=True,
    )

    for ax, (title, phi_deg) in zip(axes, locations):
        plot_slice(fig, ax, rad_dist, phi_deg, title)

    info = rad_dist.info
    fig.suptitle(
        f"{info['distType']} R={info['startR']:.2f}, z={info['startZ']:.2f} | "
        f"{path.parent.name}",
        fontsize=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"Saved {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    helical_path = args.helical_json or newest_rad_dist_json(args.root, "helical")
    elongated_path = args.elongated_json or newest_rad_dist_json(
        args.root, "elongatedRing"
    )

    plot_rad_dist(
        helical_path,
        args.output_dir / "KSTAR_helical_radDist_toroidal_slices.png",
        show=args.show,
    )
    plot_rad_dist(
        elongated_path,
        args.output_dir / "KSTAR_elongatedRing_radDist_toroidal_slices.png",
        show=args.show,
    )
