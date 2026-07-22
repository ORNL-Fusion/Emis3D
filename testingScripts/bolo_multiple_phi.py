# bolo_multiple_phi.py
"""
Test script that has a bolometer view different phi regions at once
and bin them correctly.

Skip to "HERE HERE HERE HERE HERE ..." for the start of the program, the top is just from:
https://www.cherab.info/demonstrations/bolometry/observing_radiation_function.html#bolometer-observing-radiation

See also:
https://www.cherab.info/demonstrations/line_emission/custom_emitter.html

Strategy for simultaneous region separation:
--------------------------------------------
Wavelength is used purely as a bookkeeping label. Because the scene contains
no dispersive or wavelength-dependent materials (only AbsorbingSurface), the
ray-tracing geometry is identical at every wavelength. A custom
InhomogeneousVolumeEmitter deposits emission into different spectral bins
depending on the location of the emitting point (here: above/below CENTRE_Z;
generalises trivially to toroidal phi sectors).
"""

import matplotlib.pyplot as plt
import numpy as np
from raysect.core import Node, Point3D, Vector3D, rotate_basis, translate
from raysect.optical import World
from raysect.optical.material import (
    AbsorbingSurface,  # type: ignore
    VolumeTransform,  # type: ignore
    InhomogeneousVolumeEmitter,
    NumericalIntegrator,
)
from raysect.primitive import Box, Cylinder, Subtract
from raysect.optical.material.emitter import InhomogeneousVolumeEmitter
from cherab.tools.emitters import RadiationFunction
from cherab.tools.observers.bolometry import (
    BolometerCamera,
    BolometerSlit,
    BolometerFoil,
)
from raysect.optical.observer import PowerPipeline0D, SpectralPowerPipeline0D

# Change to SerialEngine to store what points RaySect calls for each defintion
# This will not be in the Emis3D version
try:
    from raysect.core.workflow import SerialEngine
except ImportError:
    try:
        from raysect.core import SerialEngine
    except ImportError:
        SerialEngine = None
        print("WARNING: SerialEngine not importable - probes will stay empty.")

# Convenient constants
XAXIS = Vector3D(1, 0, 0)
YAXIS = Vector3D(0, 1, 0)
ZAXIS = Vector3D(0, 0, 1)
ORIGIN = Point3D(0, 0, 0)
# Bolometer geometry
BOX_WIDTH = 0.05
BOX_HEIGHT = 0.07
BOX_DEPTH = 0.2
SLIT_WIDTH = 0.004
SLIT_HEIGHT = 0.005
FOIL_WIDTH = 0.0013
FOIL_HEIGHT = 0.0038
FOIL_CORNER_CURVATURE = 0.0005
SLIT_SENSOR_SEPARATION = 0.1
FOIL_SEPARATION = 0.00508  # 0.2 inch between foils

world = World()

########################################################################
# Build a simple bolometer camera.
########################################################################

# The camera consists of a box with a rectangular slit and 4 foils.
# In its local coordinate system, the camera's slit is located at the
# origin and the foils below the z=0 plane, looking up towards the slit.
camera_box = Box(
    lower=Point3D(-BOX_WIDTH / 2, -BOX_HEIGHT / 2, -BOX_DEPTH),
    upper=Point3D(BOX_WIDTH / 2, BOX_HEIGHT / 2, 0),
)
# Hollow out the box
inside_box = Box(
    lower=camera_box.lower + Vector3D(1e-5, 1e-5, 1e-5),
    upper=camera_box.upper - Vector3D(1e-5, 1e-5, 1e-5),
)
camera_box = Subtract(camera_box, inside_box)
# The slit is a hole in the box
aperture = Box(
    lower=Point3D(-SLIT_WIDTH / 2, -SLIT_HEIGHT / 2, -1e-4),
    upper=Point3D(SLIT_WIDTH / 2, SLIT_HEIGHT / 2, 1e-4),
)
camera_box = Subtract(camera_box, aperture)
camera_box.material = AbsorbingSurface()
# Instance of the bolometer camera
bolometer_camera = BolometerCamera(camera_geometry=camera_box)
# The bolometer slit in this instance just contains targeting information
# for the ray tracing, since we have already given our camera a geometry
# The slit is defined in the local coordinate system of the camera
slit = BolometerSlit(
    slit_id="Example slit",
    centre_point=ORIGIN,
    basis_x=XAXIS,
    dx=SLIT_WIDTH,
    basis_y=YAXIS,
    dy=SLIT_HEIGHT,
    parent=bolometer_camera,
    csg_aperture=False,
)
# 4 bolometer foils, spaced at equal intervals along the local X axis
# The bolometer positions and orientations are given in the local coordinate
# system of the camera, just like the slit
sensor = Node(
    name="Bolometer sensor",
    parent=bolometer_camera,
    transform=translate(0, 0, -SLIT_SENSOR_SEPARATION),
)
# The foils are shifted relative to the centre of the sensor by -1.5, -0.5, 0.5 and 1.5
# times the foil-foil separation
for i, shift in enumerate([-1.5, -0.5, 0.5, 1.5]):
    foil_transform = translate(shift * FOIL_SEPARATION, 0, 0) * sensor.transform
    foil = BolometerFoil(
        detector_id="Foil {}".format(i + 1),
        centre_point=ORIGIN.transform(foil_transform),
        basis_x=XAXIS.transform(foil_transform),
        dx=FOIL_WIDTH,
        basis_y=YAXIS.transform(foil_transform),
        dy=FOIL_HEIGHT,
        slit=slit,
        parent=bolometer_camera,
        units="Power",
        accumulate=False,
        curvature_radius=FOIL_CORNER_CURVATURE,  # type: ignore
    )
    bolometer_camera.add_foil_detector(foil)

bolometer_camera.transform = translate(1, 0, 1.5) * rotate_basis(-ZAXIS, YAXIS)
bolometer_camera.parent = world


########################################################################
########################################################################
########################################################################
# HERE HERE HERE HERE HERE HERE HERE
########################################################################
########################################################################
########################################################################


# Used to store a history of points called by RaySect
HIST_LO, HIST_HI, HIST_N = -1.05, 0.05, 22
HIST_EDGES = np.linspace(HIST_LO, HIST_HI, HIST_N + 1)

# The plasma will be a cylindrical plasma which emits with a constant
# emissivity of 1 W/m3 in an annular ring.
# See the RadiationFunction example for more details of how this is set up.
MAJOR_RADIUS = 1
MINOR_RADIUS = 0.5
CENTRE_Z = -0.5
CYLINDER_HEIGHT = 5
EMISSIVITY = 1

N_REGIONS = 2  # top/bottom here; becomes N toroidal sectors for phi binning
WL_MIN = 300
WL_MAX = 500
STEP_SIZE = 0.005  # Integration step size
BOLO_SAMPLES = 10_000

# Probes need their own, much smaller pass. At BOLO_SAMPLES with
# STEP_SIZE=0.01 each foil would log ~4 million points per case, which is
# hundreds of MB of Python floats and unusably slow in serial mode. A few
# hundred rays is plenty to characterise WHERE samples land.
PROBE_SAMPLES = 200
PROBE_FOIL_INDEX = 1


def _new_probe():
    """Raw coordinate store, so r and z can be histogrammed afterwards."""
    return {"n": 0, "r": [], "z": []}


def _log(store, r, z):
    store["n"] += 1
    store["r"].append(r)
    store["z"].append(z)


def _reset_probes():
    for store in (probe_full, probe_func, probe_top, probe_bot):
        store["n"] = 0
        store["r"].clear()
        store["z"].clear()


probe_full = _new_probe()
probe_func = _new_probe()
probe_top = _new_probe()
probe_bot = _new_probe()

# Named registry so the probe pass and the plotting can iterate cleanly.
PROBES = {
    "full": probe_full,
    "top": probe_top,
    "bottom": probe_bot,
    "region": probe_func,
}
PROBE_ORDER = ["full", "top", "bottom", "region"]


def emission_function_full(x, y, z):
    r = np.sqrt(x**2 + y**2)
    if (r - MAJOR_RADIUS) ** 2 + (z - CENTRE_Z) ** 2 < MINOR_RADIUS**2:
        _log(probe_full, r, z)
        return EMISSIVITY
    return 0


def emission_function_top(x, y, z):
    r = np.sqrt(x**2 + y**2)
    if (r - MAJOR_RADIUS) ** 2 + (z - CENTRE_Z) ** 2 < MINOR_RADIUS**2:
        if z >= CENTRE_Z:
            _log(probe_top, r, z)
            return EMISSIVITY
    return 0


def emission_function_bottom(x, y, z):
    r = np.sqrt(x**2 + y**2)
    if (r - MAJOR_RADIUS) ** 2 + (z - CENTRE_Z) ** 2 < MINOR_RADIUS**2:
        if z < CENTRE_Z:
            _log(probe_bot, r, z)
            return EMISSIVITY
    return 0


class RegionEmitter(InhomogeneousVolumeEmitter):
    """Emitter that deposits emission into a spectral bin chosen by region.

    Wavelength acts only as a label: each spatial region is assigned one
    spectral bin. The observer must be configured with the same number of
    spectral bins over the same (WL_MIN, WL_MAX) range.

    """

    def __init__(self, n_regions=N_REGIONS, probe=False, step=None):
        # Required for InhomogeneousVolumeEmitter, see RaySectgithub source for more info
        if step is None:
            # Use default step integration size from InhomogeneousVolumeEmitter
            super().__init__()
        else:
            super().__init__(NumericalIntegrator(step=step))

        self.n_regions = n_regions
        self.probe = probe
        self.bin_width = (WL_MAX - WL_MIN) / self.n_regions

    # This requires all of these excess values, per:
    # https://www.cherab.info/demonstrations/line_emission/custom_emitter.html
    def emission_function(
        self, point, direction, spectrum, world, ray, primitive, to_local, to_world
    ):

        x, y, z = point.transform(to_world)
        r = np.sqrt(x**2 + y**2)

        # Outside the annular plasma ring: no emission
        if (r - MAJOR_RADIUS) ** 2 + (z - CENTRE_Z) ** 2 > MINOR_RADIUS**2:
            return spectrum

        # Add value to the probe
        if self.probe:
            _log(probe_func, r, z)

        # Region 0 = bottom, region 1 = top
        region = 1 if z >= CENTRE_Z else 0

        # Determine what wavelength to store
        tag_wavelength = WL_MIN + (region + 0.5) * self.bin_width
        index = int(
            (tag_wavelength - spectrum.min_wavelength) / spectrum.delta_wavelength
        )
        if 0 <= index < spectrum.bins:
            # Deposit the full emissivity into a single bin as spectral
            # radiance [W/m3/str/nm]. Dividing by delta_wavelength means
            # the wavelength-integrated radiance of this bin recovers
            # EMISSIVITY / 4pi, matching what RadiationFunction produces
            # spread over the whole spectrum.
            spectrum.samples[index] += EMISSIVITY / (
                4 * np.pi * spectrum.delta_wavelength
            )
        return spectrum


emitter = Cylinder(
    radius=MAJOR_RADIUS + MINOR_RADIUS,
    height=CYLINDER_HEIGHT,
    transform=translate(0, 0, CENTRE_Z - CYLINDER_HEIGHT / 2),
    parent=world,
)

########################################################################
# 1. Measure everything at once as a sanity check
########################################################################
emitter.material = VolumeTransform(
    RadiationFunction(emission_function_full, step=STEP_SIZE),
    transform=emitter.transform.inverse(),
)
data = {}
data["SANITY CHECK"] = []
for foil in bolometer_camera:
    foil.pixel_samples = BOLO_SAMPLES
    foil.units = "Power"
    foil.observe()
    data["SANITY CHECK"].append(foil.pipelines[0].value.mean)


########################################################################
# 2. Measure the top and bottom separately (two independent runs)
########################################################################
emitter.material = VolumeTransform(
    RadiationFunction(emission_function_top, step=STEP_SIZE),
    transform=emitter.transform.inverse(),
)

data["TOP ONLY"] = []
for foil in bolometer_camera:
    foil.pixel_samples = BOLO_SAMPLES
    foil.units = "Power"
    foil.observe()
    data["TOP ONLY"].append(foil.pipelines[0].value.mean)


emitter.material = VolumeTransform(
    RadiationFunction(emission_function_bottom, step=STEP_SIZE),
    transform=emitter.transform.inverse(),
)

data["BOTTOM ONLY"] = []
for foil in bolometer_camera:
    foil.pixel_samples = BOLO_SAMPLES
    foil.units = "Power"
    foil.observe()
    data["BOTTOM ONLY"].append(foil.pipelines[0].value.mean)


########################################################################
# 3. Measure both at the same time, using wavelength to differentiate
#    top and bottom. One observe() per foil, region-resolved output.
########################################################################

# Note: RegionEmitter works in world coordinates internally
# (point.transform(to_world)), so no VolumeTransform wrapper is needed.
emitter.material = RegionEmitter(n_regions=N_REGIONS, step=STEP_SIZE)

data["SPECTRAL BOTTOM"] = []  # region 0 -> bin 0 (300-400 nm)
data["SPECTRAL TOP"] = []  # region 1 -> bin 1 (400-500 nm)
data["SPECTRAL TOTAL"] = []
data["SPECTRAL CHECK"] = []

bin_width = (WL_MAX - WL_MIN) / N_REGIONS

for foil in bolometer_camera:
    # Replace the power pipeline with a spectral one. accumulate=False to
    # match the behaviour of the earlier measurements.
    power_pipeline = PowerPipeline0D(accumulate=False)
    spectral_pipeline = SpectralPowerPipeline0D(
        display_progress=False, accumulate=False
    )
    foil.pipelines = [spectral_pipeline, power_pipeline]

    # The observer's spectral sampling must match the emitter's binning
    foil.min_wavelength = WL_MIN
    foil.max_wavelength = WL_MAX
    foil.spectral_bins = N_REGIONS

    foil.pixel_samples = BOLO_SAMPLES
    foil.observe()

    # samples.mean is spectral power [W/nm] in each bin; multiply by the
    # bin width to get power [W] per region.
    wavelengths = spectral_pipeline.wavelengths
    delta = wavelengths[1] - wavelengths[0]
    spectral_power = spectral_pipeline.samples.mean  # shape (N_REGIONS,)

    edges = np.linspace(WL_MIN, WL_MAX, N_REGIONS + 1)

    region_powers = np.array(
        [
            spectral_power[(wavelengths >= lo) & (wavelengths < hi)].sum() * delta
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )
    data["SPECTRAL CHECK"].append(power_pipeline.value.mean)
    data["SPECTRAL BOTTOM"].append(region_powers[0])
    data["SPECTRAL TOP"].append(region_powers[1])
    data["SPECTRAL TOTAL"].append(region_powers.sum())


########################################################################
# Print the results
########################################################################
header = (
    f"{'Foil':>4} | {'Sanity':>10} | {'Top+Bot':>10} | "
    f"{'Spec total':>10} | {'Spec Check':>10} | {'Spec top':>10} | {'Top only':>10} | "
    f"{'Spec bot':>10} | {'Bot only':>10}"
)
print(header)
print("-" * len(header))
for ii in range(4):
    tb = data["TOP ONLY"][ii] + data["BOTTOM ONLY"][ii]
    print(
        f"{ii + 1:>4} | {data['SANITY CHECK'][ii]:>10.3e} | {tb:>10.3e} | "
        f"{data['SPECTRAL TOTAL'][ii]:>10.3e} | "
        f"{data['SPECTRAL CHECK'][ii]:>10.3e} | "
        f"{data['SPECTRAL TOP'][ii]:>10.3e} | {data['TOP ONLY'][ii]:>10.3e} | "
        f"{data['SPECTRAL BOTTOM'][ii]:>10.3e} | {data['BOTTOM ONLY'][ii]:>10.3e}"
    )


########################################################################
# PROBE PASS: where does each emitter actually deposit emission?
########################################################################
# Run LAST so it cannot disturb the measurements above. Three requirements,
# all of which must hold or the probes record nothing:
#   1. SerialEngine  - otherwise the emission function runs in a worker
#                      process and its appends are thrown away.
#   2. probe=True    - RegionEmitter logs nothing by default.
#   3. small samples - see PROBE_SAMPLES.
print("\n" + "=" * 70)
print("PROBE PASS")
print("=" * 70)

if SerialEngine is None:
    print("SKIPPED: SerialEngine unavailable.")
else:
    probe_foil = list(bolometer_camera)[PROBE_FOIL_INDEX]

    # Save state we are about to trample
    _saved_samples = probe_foil.pixel_samples
    try:
        _saved_engine = probe_foil.render_engine
    except AttributeError:
        _saved_engine = None

    _reset_probes()

    probe_configs = [
        (
            "full",
            VolumeTransform(
                RadiationFunction(emission_function_full, step=STEP_SIZE),
                transform=emitter.transform.inverse(),
            ),
        ),
        (
            "top",
            VolumeTransform(
                RadiationFunction(emission_function_top, step=STEP_SIZE),
                transform=emitter.transform.inverse(),
            ),
        ),
        (
            "bottom",
            VolumeTransform(
                RadiationFunction(emission_function_bottom, step=STEP_SIZE),
                transform=emitter.transform.inverse(),
            ),
        ),
        # probe=True is essential here
        ("region", RegionEmitter(n_regions=N_REGIONS, step=STEP_SIZE, probe=True)),
    ]

    for name, material in probe_configs:
        emitter.material = material
        probe_foil.units = "Power"  # units setter rebuilds pipelines
        probe_foil.min_wavelength = WL_MIN
        probe_foil.max_wavelength = WL_MAX
        probe_foil.spectral_bins = N_REGIONS
        probe_foil.pixel_samples = PROBE_SAMPLES
        probe_foil.render_engine = SerialEngine()
        probe_foil.observe()
        print(f"  {name:>8}: logged {len(PROBES[name]['z']):>8} points")

    # Restore
    probe_foil.pixel_samples = _saved_samples
    if _saved_engine is not None:
        probe_foil.render_engine = _saved_engine


########################################################################
# Probe histograms: r and z for each case
########################################################################
def _describe(name, store):
    if store["n"] == 0:
        print(f"  {name:>8}: EMPTY")
        return
    z = np.asarray(store["z"])
    r = np.asarray(store["r"])
    print(
        f"  {name:>8}: n={store['n']:>8}  "
        f"z[{z.min():+.4f}, {z.max():+.4f}]  "
        f"r[{r.min():.4f}, {r.max():.4f}]  "
        f"frac(z>=CENTRE_Z)={np.mean(z >= CENTRE_Z):.4f}"
    )


print("\nProbe summary:")
for _name in PROBE_ORDER:
    _describe(_name, PROBES[_name])

_populated = [n for n in PROBE_ORDER if PROBES[n]["n"] > 0]
if _populated:
    fig2, axes2 = plt.subplots(
        2, len(_populated), figsize=(4 * len(_populated), 7), squeeze=False
    )
    # Common bins so the four cases are directly comparable
    z_bins = np.linspace(-1.05, 0.05, 45)
    r_bins = np.linspace(0.80, 1.05, 45)

    for col, name in enumerate(_populated):
        z = np.asarray(PROBES[name]["z"])
        r = np.asarray(PROBES[name]["r"])

        ax = axes2[0][col]
        ax.hist(z, bins=z_bins, color="tab:blue")
        ax.axvline(CENTRE_Z, color="k", ls="--", lw=1, label="CENTRE_Z")
        ax.set_title(f"{name}  (n={len(z)})")
        ax.set_xlabel("z [m]")
        if col == 0:
            ax.set_ylabel("samples per z bin")
            ax.legend(fontsize=8)

        ax = axes2[1][col]
        ax.hist(r, bins=r_bins, color="tab:orange")
        ax.set_xlabel("r [m]")
        if col == 0:
            ax.set_ylabel("samples per r bin")

    fig2.suptitle(
        "Where each emitter deposits: sample distribution in z (top) and r "
        "(bottom)\n"
        "Gaps or every-other-bin structure in z means the integration step "
        "is too coarse"
    )
    fig2.tight_layout()
    fig2.savefig("probe_histograms.png", dpi=130)
    print("\nSaved probe_histograms.png")
else:
    print("\nNo probe recorded any samples - nothing to plot.")


########################################################################
# Nice figure
########################################################################
foil_ids = np.arange(1, 5)
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

axes[0].bar(foil_ids - width / 2, data["TOP ONLY"], width, label="Top (separate run)")
axes[0].bar(foil_ids + width / 2, data["SPECTRAL TOP"], width, label="Top (region run)")
axes[0].set_title("Top region")
axes[0].set_xlabel("Foil")
axes[0].set_ylabel("Power [W]")
axes[0].legend()

axes[1].bar(
    foil_ids - width / 2, data["BOTTOM ONLY"], width, label="Bottom (separate run)"
)
axes[1].bar(
    foil_ids + width / 2,
    data["SPECTRAL BOTTOM"],
    width,
    label="Bottom (region run)",
)
axes[1].set_title("Bottom region")
axes[1].set_xlabel("Foil")
axes[1].legend()

fig.suptitle("Region separation: independent runs vs single wavelength-tagged run")
fig.tight_layout()
plt.show()
