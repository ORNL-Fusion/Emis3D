"""
tests/test_toroidal_floor.py

A toroidally uniform floor under the peaked profile (toroidal_floor: true):
a * g(dphi) + c with c >= 0, for the exponential and linear scales, in both
solvers; and the check that the cameras sample enough points of the
profile to determine its free parameters.
"""

import numpy as np
import pytest

import emis3d.uncertainty as U
from emis3d import util_emis3D as Util_emis3D
from emis3d.radDistFitting import RadDistFitting
from emis3d.run_config import RunConfig

INJ = 225.0
N_CH, N_PHI = 10, 360
SPREAD = {"CAM_A": 135.0, "CAM_B": 270.0, "CAM_C": 45.0}     # 90, 45 and 180 deg from the injector: three points


def test_scale_functions_add_the_floor():
    dphi = np.array([0.0, 1.0, np.pi])
    np.testing.assert_allclose(Util_emis3D.scale_exp(2.0, 0.5, dphi, 0.3), 2.0 * np.exp(-0.5 * dphi**2) + 0.3)
    np.testing.assert_allclose(Util_emis3D.scale_linear(2.0, 1.0, dphi, 0.3), np.maximum(2.0 - dphi, 0.0) + 0.3)
    np.testing.assert_allclose(Util_emis3D.scale_exp(2.0, 0.5, dphi), 2.0 * np.exp(-0.5 * dphi**2))       # default: no floor
    w = Util_emis3D.scale_wrapper(a=1.0, b=1.0, phi=np.deg2rad([225.0, 45.0]), mu=np.deg2rad(INJ), scale_def="exponential",
                                  emissionName="ring", c=0.25)
    assert w[0] == pytest.approx(1.25) and w[1] == pytest.approx(np.exp(-np.pi**2) + 0.25)


# ---------------------------------------------------------------------------
# The library solver
# ---------------------------------------------------------------------------

def candidate(groups):
    rd = RadDistFitting()
    shape = np.exp(-0.5 * ((np.arange(N_CH) - 4.5) / 2.0) ** 2) + 0.05
    phi = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    rd.info.update({"units": "Brightness", "emissionNames": ["elongatedRing"], "injectionLocation": INJ, "distType": "elongatedRing"})
    data = {"Brightness": {"channelOrder": {}, "elongatedRing": {}}, "Brightness_error": {"elongatedRing": {}},
            "observed_phi_loc": {"elongatedRing": {}},
            "toroidalRadiatedPower": {"elongatedRing": {"phi_array": phi, "P_pol": np.ones(N_PHI)}}}
    for g, deg in groups.items():
        data["Brightness"]["channelOrder"][g] = [f"{g}{i}" for i in range(N_CH)]
        data["Brightness"]["elongatedRing"][g] = [[v] for v in shape]
        data["Brightness_error"]["elongatedRing"][g] = [[0.0] for _ in shape]
        data["observed_phi_loc"]["elongatedRing"][g] = [[np.deg2rad(deg)] for _ in shape]
    rd.data = data
    rd._load_ok = True
    rd._map_signals()
    return rd, shape


def floor_data(groups, shape, b, floor, amp=1000.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    obs, err = [], []
    for deg in groups.values():
        dphi = Util_emis3D.find_dphi(np.array([np.deg2rad(deg)]), np.deg2rad(INJ), "elongatedRing")[0]
        clean = amp * (np.exp(-b * dphi**2) + floor) * shape
        sig = noise * amp * (1.0 + floor) * np.ones_like(clean)
        obs.append(list(clean + rng.normal(0, sig)))
        err.append(list(sig))
    return {"observed": obs, "observed_error": err}


def library_fit(groups, data, floor):
    rd, _ = candidate(groups)
    rd.prepare_for_fits([[f"{g}{i}" for i in range(N_CH)] for g in groups], data_max=max(max(v) for v in data["observed"]))
    rd.create_parameters(enable_dphi_scaling=True, scale_def="exponential", toroidal_floor=floor)
    synth = {"info": rd.info, "injectionLocation": INJ, "injectionLocation_rad": np.deg2rad(INJ), "emissionNames": ["elongatedRing"]}
    synth["elongatedRing"] = dict(rd.fitSynthetic["elongatedRing"])
    synth["elongatedRing"]["phi_array"] = rd.data["toroidalRadiatedPower"]["elongatedRing"]["phi_array"]
    synth["elongatedRing"]["P_pol"] = rd.data["toroidalRadiatedPower"]["elongatedRing"]["P_pol"]
    fit = Util_emis3D.fit_radDist(rd.fitSynthetic["params"]["params"], data, synth, "exponential", 0.0, method="least_squares")
    return fit, synth


def test_library_solver_recovers_a_floor_and_its_profile():
    _, shape = candidate(SPREAD)
    data = floor_data(SPREAD, shape, b=1.0, floor=0.2)
    fit, synth = library_fit(SPREAD, data, floor=True)
    p = fit.params.valuesdict()
    assert p["c_225"] / p["a_225"] == pytest.approx(0.2, abs=0.02)
    assert p["b_elongatedRing_225"] == pytest.approx(1.0, abs=0.1)
    q, P = U.derived_quantities(p, synth, "exponential")
    assert q["floor_fraction"] == pytest.approx(0.2, abs=0.02)
    phi = synth["elongatedRing"]["phi_array"]
    true_P = np.exp(-1.0 * Util_emis3D.find_dphi(phi, np.deg2rad(INJ), "elongatedRing") ** 2) + 0.2
    assert q["tpf"] == pytest.approx(true_P.max() / true_P.mean(), rel=0.05)              # the floor lowers the TPF
    without, _ = library_fit(SPREAD, data, floor=False)
    assert without.redchi > 5 * fit.redchi                                                  # a peak alone cannot fit it


def test_library_floor_vanishes_when_there_is_none():
    _, shape = candidate(SPREAD)
    fit, _ = library_fit(SPREAD, floor_data(SPREAD, shape, b=1.0, floor=0.0), floor=True)
    p = fit.params.valuesdict()
    assert p["c_225"] / p["a_225"] < 0.02


# ---------------------------------------------------------------------------
# Can the cameras determine it?
# ---------------------------------------------------------------------------

BASE = {"BOLOMETERS": {"DISRADU": {"dataFileName": "a"}, "SX90MF": {"dataFileName": "b"}, "SX90PF": {"dataFileName": "c"}},
        "shot": 1, "radDistDirectories_LocIndependent": ["x"], "scale_def": "exponential", "enable_dphi_scaling": True,
        "injectionLocation": 225.0}
DIIID = {"DISRADU": 135.0, "SX90MF": 270.0, "SX90PF": 270.0}


def test_points_sampled_count_distances_from_the_injector():
    cfg = RunConfig.from_dict(BASE)
    assert cfg.toroidal_points(DIIID, 225.0) == [45.0, 90.0]                                  # two points: 45 and 90 deg away
    assert cfg.toroidal_identifiability(DIIID) is None                                        # a and b: determined
    floor = RunConfig.from_dict({**BASE, "toroidal_floor": True})
    msg = floor.toroidal_identifiability(DIIID)
    assert msg and "3 free parameters" in msg and "only 2 point(s)" in msg
    sym = {"DISRADU": 135.0, "SX90MF": 315.0, "SX90PF": 270.0}                                # 135 and 315 are both 90 deg away
    assert RunConfig.from_dict({**BASE, "toroidal_floor": True}).toroidal_points(sym, 225.0) == [45.0, 90.0]
    moving = RunConfig.from_dict({**BASE, "vary_peak_rad_location": True})
    assert moving.toroidal_points(sym, 225.0) == [135.0, 270.0, 315.0] and moving.toroidal_identifiability(sym) is None


def test_floor_config_validation():
    assert RunConfig.from_dict(BASE).toroidal_floor is False
    assert RunConfig.from_dict({**BASE, "scale_def": "linear", "toroidal_floor": True}).toroidal_unknowns() == ["a", "b", "c (floor)"]
    for scale in ("constant", "step"):
        with pytest.raises(ValueError, match="toroidal_floor needs scale_def"):
            RunConfig.from_dict({**BASE, "scale_def": scale, "toroidal_floor": True})
    with pytest.raises(ValueError, match="true or false"):
        RunConfig.from_dict({**BASE, "toroidal_floor": "yes"})


def test_check_and_a_library_fit_with_the_floor(tmp_path):
    from emis3d.emis3D import Emis3D
    from emis3d.pipeline import check
    from test_calibration import CAMS, write_workspace

    run = write_workspace(tmp_path, tstart=10.0, miscal={c: 1.0 for c in CAMS}, scale_def="exponential",
                          enable_dphi_scaling=True, toroidal_floor=True)
    msgs = [f.message for f in check(run)]
    assert any("toroidal profile" in m for m in msgs)
    e = Emis3D(runConfigName=str(run))
    e._prepare_fits(evalTime=10.0)
    p = e.fits[10.0][0]["parameters"]
    assert "c_0" in p and p["c_0"].min == 0.0                                                # the floor, free to vanish
    e._minimize_radDists(evalTime=10.0)
    assert any(f.get("fit") is not None and f["fit"].params["c_0"].value >= 0.0 for f in e.fits[10.0].values())
