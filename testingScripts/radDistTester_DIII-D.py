# radDistTester_DIII-D.py
"""
This program will group similar SXR arrays, then plot out
the chords, radDist contour plot, and the observed radiation
below.

It is currently specific to DIII-D
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_REPO_ROOT))

import numpy as np
import main.Util_radDist as Util_radDist
from main.Globals import EMIS3D_INPUTS_DIRECTORY
from main.Util import config_loader
import matplotlib.pyplot as plt
from scipy.integrate import simpson

tokamakName = "DIII-D"
configFileName = "helical_config.yaml"  # "sqaureTube_config.yaml"  # "elongatedRing_config.yaml"  # "helical_config.yaml"
rzvalues = [2.029, 0.409]

plt.ion()

# --- Create the radDist using only one point, we don't need to loop over everything
pathFileName = EMIS3D_INPUTS_DIRECTORY / tokamakName / "radDists" / configFileName

config = config_loader(pathFileName)
if config is None:
    raise FileNotFoundError(f"Could not load config file: {pathFileName}")


# --- Update the configuration file
rzArray = np.array([rzvalues[0], rzvalues[1]])

if "sigma_R_vals" in config:
    config["sigma_R"] = config["sigma_R_vals"][0]
if "sigma_z_vals" in config:
    config["sigma_z"] = config["sigma_z_vals"][0]
if "rotationAngles" in config:
    config["rotationAngle"] = config["rotationAngles"][0]
arg_list = (rzArray, config)


"""
# --- Try to identify bottlenecks
t0 = time.time()
rD = radDist.Helical(startR=rzArray[0], startZ=rzArray[1], config=config)
print(f"Initilization: {time.time() - t0:.1f} s")

# --- Building the tokamak
t0 = time.time()
rD._build_tokamak(
    tokamakName=rD.info["tokamakName"],
    mode="Build",
    reflections=False,
    eqFileName=rD.info["eqFileName"],
)
print(f"Built tokamak: {time.time() - t0:.1f} s")

# --- Setting the field lines
t0 = time.time()
rD.setFieldLine()
print(f"Field lines created: {time.time() - t0:.1f} s")


# --- Going through rD.build()
t0 = time.time()
# rD.power_per_bin_calc()
print(f"Power per bin caclulated: {time.time() - t0:.1f} s")

t0 = time.time()
rD.data = {}
rD.bolos_observe()

print(f"Bolos observed in: {time.time() - t0:.1f} s")

"""


# --- Create the radDist
if config["distType"] == "Helical":
    rD = Util_radDist.radDist_Helical_parallel(arg_list, return_result=True)
elif config["distType"] == "HelicalRing":
    rD = Util_radDist.radDist_HelicalRing_parallel(arg_list, return_result=True)
elif config["distType"] == "ElongatedRing":
    rD = Util_radDist.radDist_ElongatedRing_parallel(arg_list, return_result=True)
elif config["distType"] == "SquareTube":
    rD = Util_radDist.radDist_SquareTube_parallel(arg_list, return_result=True)
else:
    raise RuntimeError(
        "Please have 'elongatedRing', 'helical', 'HelicalRing', or 'SqureTube' in the configFileName"
    )


# --- Plot everything ---
if rD is not None:
    rD.plotOverview(plot_etendue=["SX45F07"])


# --- Testing out a new powerPerBin calculation, utilizing simposon integration instead of a monte carlo method
def total_radiated_power(
    # self,
    rD,
    n_phi: int = 100,
    n_poloidal: int = 200,
    # emissionName: str | None = None,
) -> dict:
    """
    Compute total radiated power by deterministic quadrature.

    Evaluates the poloidal integral P_pol(φ) at each toroidal location:
        P_pol(φ) = ∫∫_wall ε(R, z, φ) · R dR dz


    Parameters
    ----------
    n_phi      : int — number of toroidal quadrature points.
    n_poloidal : int — number of points along each poloidal axis.
    emissionName : str, optional — defaults to self.emissionName.

    Returns
    -------
    dict with keys:
        phi_array : (n_phi,)  toroidal angles evaluated (radians).
        P_pol     : (n_phi,)  poloidal integral at each phi.
        P_total   : float     total radiated power assuming toroidal symmetry (W).
    """

    wall = rD.tokamak.wall

    emissionNames = rD.info["emissionNames"]

    # -- Full grid creation
    R_vals = np.linspace(wall["minr"], wall["maxr"], n_poloidal)
    z_vals = np.linspace(wall["minz"], wall["maxz"], n_poloidal)
    RR, ZZ = np.meshgrid(R_vals, z_vals, indexing="ij")
    R_flat = RR.ravel()
    z_flat = ZZ.ravel()

    phi_array = np.linspace(0, 2.0 * np.pi, n_phi, endpoint=False)

    # --- Checking area/volume values
    if False:
        full_grid = np.column_stack([RR.ravel(), ZZ.ravel()])
        inside = rD.tokamak._inside_tokamak(full_grid)
        # Cross-sectional area: ∫∫_wall dR dz
        area_grid = np.zeros(len(R_flat))
        area_grid[inside] = 1.0
        area_grid = area_grid.reshape(n_poloidal, n_poloidal)
        cross_section_area = simpson(simpson(area_grid, x=z_vals, axis=1), x=R_vals)

        # Cylindrical volume: 2π ∫∫_wall R dR dz
        vol_grid = np.zeros(len(R_flat))
        vol_grid[inside] = R_flat[inside]
        vol_grid = vol_grid.reshape(n_poloidal, n_poloidal)
        volume = 2.0 * np.pi * simpson(simpson(vol_grid, x=z_vals, axis=1), x=R_vals)

        bbox_area = (R_vals.max() - R_vals.min()) * (z_vals.max() - z_vals.min())
        cfg_volume = rD.tokamak.info["MACHINE"]["volume"]
        vol_ratio = cfg_volume / volume

        print("── Volume checks ────────────────────────────────────────")
        print(f"  Bounding box area      : {bbox_area:.4f} m²")
        print(
            f"  Cross-sectional area   : {cross_section_area:.4f} m²  "
            f"({'<' if cross_section_area < bbox_area else '>'} bounding box ✓)"
            if cross_section_area < bbox_area
            else "  ⚠ cross-section exceeds bounding box"
        )
        print(f"  Cylindrical volume     : {volume:.4f} m³")
        print(f"  Config volume          : {cfg_volume:.4f} m³")
        print(
            f"  Ratio (config/computed): {vol_ratio:.4f} "
            f"{'✓' if np.isclose(vol_ratio, 1.0, rtol=0.05) else '⚠ mismatch'}"
        )
        print("─────────────────────────────────────────────────────────")

    # --- Poloidal integral P_pol(φ) at each toroidal location
    P_pol = np.zeros(n_phi)

    emissionName = emissionNames[0]

    for i, phi in enumerate(phi_array):

        phi_arr = np.full(len(R_flat), phi)

        emiss = np.zeros(len(R_flat))
        for emissionName in emissionNames:
            # --- Caclulate the emissivity at that toroidal location
            result = rD.calc_emissivity(
                R_flat, z_flat, phi_arr, emissionName=emissionName
            )
            emiss += result[emissionName]  # (N_inside,)

        # Reshape grid for integration
        emis_grid = (emiss * R_flat).reshape(n_poloidal, n_poloidal)

        inner = simpson(emis_grid, x=z_vals, axis=1)  # (n_poloidal,)
        P_pol[i] = simpson(inner, x=R_vals)

    # ── 4. Total power assuming toroidal symmetry ─────────────────────────────
    P_total = simpson(P_pol, x=phi_array)

    return {
        "phi_array": phi_array,
        "P_pol": P_pol,
        "P_total": P_total,
    }


if False:
    print(f"Caculating total power using 100 phi bins")
    t0 = time.time()
    if rD is not None:
        rD.power_per_bin_calc()
    print(f"Power per bin time: {time.time() - t0:.2f} s")

    t0 = time.time()
    ans = total_radiated_power(rD)
    print(f"New integration method: {time.time() - t0:.2f} s")

    # --- Plot out the results

    k_ = list(rD.data["powerPerBin"].keys())
    ppB = np.zeros(len(rD.data["powerPerBin"][k_[0]]))
    for k in k_:
        ppB += rD.data["powerPerBin"][k_[0]]

    """
    f = plt.figure()
    ax = f.add_subplot(111)

    n_phi = len(rD.data["powerPerBin"][k_[0]])
    ax.plot(
        np.linspace(0, 2.0 * np.pi, n_phi),
        ppB,
        color="black",
        label="powerPerBin",
    )
    ax.plot(ans["phi_array"], ans["P_pol"], color="purple", label="Integration method")
    ax.legend()
    ax.set_xlabel("phi [rad]")
    plt.tight_layout()
    plt.show()
    """

    print(f"Total radiated power (old method): {ppB.sum():.2f}")
    print(f"Total radiated power (new method): {ans['P_total']:.2f}")
    print(f"Difference: {ppB.sum() - ans['P_total']:.2f}")


# Check 2: does power_per_bin give total_volume?
# If MC gives total_volume for uniform emissivity, the difference
# must be in how non-uniform ε interacts with the sampling measure.
