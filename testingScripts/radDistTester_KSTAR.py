#!/usr/bin/env python3
# radDistTester_KSTAR.py
"""
Create one KSTAR radDist from a config and plot the overview diagnostic.

This is a manual/interactive tester, similar to the DIII-D and JET versions. It
builds one radiation distribution at a selected R, z location so the bolometer
chords, radDist contour, injection-location contour, and synthetic channel
signals can be inspected before generating a full radDist grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_REPO_ROOT))


TOKAMAK_NAME = "KSTAR"
DEFAULT_DIST_TYPE = "elongatedRing"
DEFAULT_CONFIGS = {
    "elongatedring": "elongatedRing_config_KSTAR.yaml",
    "helical": "helical_config_KSTAR_26525.yaml",
    "newhelical": "helical_config_KSTAR_26525.yaml",
}
DIST_TYPE_PRESETS = {
    "newhelical": {
        "distType": "Helical",
        "sigma_R": 0.06,
        "sigma_z": 0.05,
        "sigmaKernel": 0.05,
        "numFieldLines": 2_000,
    },
}
DEFAULT_RZ = (1.79, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and plot a single KSTAR radDist for quick visual inspection."
        )
    )
    parser.add_argument(
        "--dist-type",
        choices=["elongatedRing", "helical", "newHelical"],
        default=DEFAULT_DIST_TYPE,
        help=(
            "KSTAR radDist type to test when --config is not supplied. "
            "Defaults to elongatedRing."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Config file under inputs/KSTAR/radDists, or an absolute/relative "
            "path to a YAML config. Overrides --dist-type."
        ),
    )
    parser.add_argument(
        "--r",
        type=float,
        default=DEFAULT_RZ[0],
        help="Starting major radius for the test radDist [m].",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=DEFAULT_RZ[1],
        help="Starting vertical location for the test radDist [m].",
    )
    parser.add_argument(
        "--sigma-r",
        type=float,
        default=None,
        help="Override sigma_R. Defaults to the first sigma_R_vals entry.",
    )
    parser.add_argument(
        "--sigma-z",
        type=float,
        default=None,
        help="Override sigma_z. Defaults to the first sigma_z_vals entry.",
    )
    parser.add_argument(
        "--rotation-angle",
        type=float,
        default=None,
        help="Override rotationAngle. Defaults to the first rotationAngles entry.",
    )
    parser.add_argument(
        "--sigma-kernel",
        type=float,
        default=None,
        help="Override sigmaKernel for helical radDists.",
    )
    parser.add_argument(
        "--num-field-lines",
        type=int,
        default=None,
        help="Override numFieldLines for helical radDists.",
    )
    parser.add_argument(
        "--plot-etendue",
        nargs="*",
        default=[],
        help="Optional channel names to pass through to plotOverview(plot_etendue=...).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the overview figure instead of only showing it.",
    )
    return parser.parse_args()


def normalize_dist_type(dist_type: str) -> str:
    return dist_type.replace("_", "").replace("-", "").lower()


def default_config_name(dist_type: str) -> str:
    key = normalize_dist_type(dist_type)
    try:
        return DEFAULT_CONFIGS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(DEFAULT_CONFIGS))
        raise ValueError(
            f"Unsupported KSTAR dist type {dist_type!r}; use one of {supported}"
        ) from exc


def resolve_config_path(config_name: str | None, dist_type: str) -> Path:
    from main.Globals import EMIS3D_INPUTS_DIRECTORY

    if config_name is None:
        config_name = default_config_name(dist_type)

    config_path = Path(config_name)
    if config_path.is_absolute() or config_path.exists():
        return config_path
    return EMIS3D_INPUTS_DIRECTORY / TOKAMAK_NAME / "radDists" / config_name


def apply_single_value_overrides(config: dict, args: argparse.Namespace) -> None:
    if args.config is None:
        config.update(DIST_TYPE_PRESETS.get(normalize_dist_type(args.dist_type), {}))

    if args.sigma_r is not None:
        config["sigma_R"] = args.sigma_r
    elif "sigma_R" not in config and "sigma_R_vals" in config:
        config["sigma_R"] = config["sigma_R_vals"][0]

    if args.sigma_z is not None:
        config["sigma_z"] = args.sigma_z
    elif "sigma_z" not in config and "sigma_z_vals" in config:
        config["sigma_z"] = config["sigma_z_vals"][0]

    if args.rotation_angle is not None:
        config["rotationAngle"] = args.rotation_angle
    elif "rotationAngle" not in config and "rotationAngles" in config:
        config["rotationAngle"] = config["rotationAngles"][0]

    if args.sigma_kernel is not None:
        config["sigmaKernel"] = args.sigma_kernel

    if args.num_field_lines is not None:
        config["numFieldLines"] = args.num_field_lines


def build_rad_dist(rz_array: np.ndarray, config: dict):
    import main.Util_radDist as Util_radDist

    arg_list = (rz_array, config)
    dist_type = normalize_dist_type(config["distType"])

    if dist_type == "helical":
        return Util_radDist.radDist_Helical_parallel(arg_list, return_result=True)
    if dist_type == "helicalring":
        return Util_radDist.radDist_HelicalRing_parallel(arg_list, return_result=True)
    if dist_type == "elongatedring":
        return Util_radDist.radDist_ElongatedRing_parallel(arg_list, return_result=True)
    if dist_type == "squaretube":
        return Util_radDist.radDist_SquareTube_parallel(arg_list, return_result=True)

    raise RuntimeError(
        "Please set config['distType'] to Helical, HelicalRing, ElongatedRing, "
        "or SquareTube."
    )


def main() -> None:
    args = parse_args()

    if args.save is not None:
        import matplotlib

        matplotlib.use("Agg")

    from main.Util import config_loader

    config_path = resolve_config_path(args.config, args.dist_type)
    config = config_loader(config_path)
    if config is None:
        raise FileNotFoundError(f"Could not load config file: {config_path}")

    if config.get("tokamakName") != TOKAMAK_NAME:
        raise ValueError(
            f"Expected a {TOKAMAK_NAME} config, got tokamakName={config.get('tokamakName')!r}"
        )

    apply_single_value_overrides(config, args)
    rz_array = np.array([args.r, args.z])

    print(f"Building {config['distType']} KSTAR radDist from {config_path}")
    print(f"R={rz_array[0]:.3f} m, z={rz_array[1]:.3f} m")

    rad_dist = build_rad_dist(rz_array, config)
    if rad_dist is None:
        raise RuntimeError("radDist creation returned None")

    if args.save is not None:
        fig = rad_dist.plotOverview(
            return_figure=True,
            plot_etendue=args.plot_etendue,
        )
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=160)
        print(f"Saved {args.save}")
    else:
        rad_dist.plotOverview(plot_etendue=args.plot_etendue)


if __name__ == "__main__":
    main()
