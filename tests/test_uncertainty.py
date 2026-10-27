"""
Tests for emis3d.uncertainty (review issue 5 / Phase 6).

Fake radDists are built through the real RadDistFitting class (no ray tracing):
each bolometer group views one toroidal plane, and the emissivity's toroidal
shape comes only from the fitted scale function, so the true TPF is known.
"""

import numpy as np
import pytest
import yaml

import emis3d.util_emis3D as Util_emis3D
from emis3d import uncertainty as U
from emis3d.radDistFitting import RadDistFitting
from emis3d.run_config import RunConfig

INJ_DEG = 225.0
GROUPS = {"CAM_A": 135.0, "CAM_B": 270.0, "CAM_C": 315.0}   # camera phi [deg]
N_CH = 12
N_PHI = 100


def poloidal_shape(kind: str, n=N_CH):
    x = np.linspace(-1, 1, n)
    if kind == "narrow":
        return np.exp(-x**2 / (2 * 0.3**2)) + 0.05
    if kind == "wide":
        return np.exp(-x**2 / (2 * 0.8**2)) + 0.05
    if kind == "shifted":
        return np.exp(-(x - 0.4)**2 / (2 * 0.4**2)) + 0.05
    raise ValueError(kind)


def fake_raddist(kind="narrow", groups=GROUPS, P_pol=None):
    """A RadDistFitting object with the same structure as a loaded radDist JSON."""
    rd = RadDistFitting()
    shape = poloidal_shape(kind)
    phi = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    rd.info.update({"units": "Brightness", "emissionNames": ["elongatedRing"],
                    "injectionLocation": INJ_DEG, "distType": "elongatedRing"})
    data = {"Brightness": {"channelOrder": {}, "elongatedRing": {}},
            "Brightness_error": {"elongatedRing": {}},
            "observed_phi_loc": {"elongatedRing": {}},
            "toroidalRadiatedPower": {"elongatedRing": {
                "phi_array": phi, "P_pol": np.ones(N_PHI) if P_pol is None else P_pol}}}
    for g, phi_deg in groups.items():
        chans = [f"{g}{i}" for i in range(N_CH)]
        data["Brightness"]["channelOrder"][g] = chans
        data["Brightness"]["elongatedRing"][g] = [[v] for v in shape]
        data["Brightness_error"]["elongatedRing"][g] = [[0.0] for _ in shape]
        data["observed_phi_loc"]["elongatedRing"][g] = [[np.deg2rad(phi_deg)] for _ in shape]
    rd.data = data
    rd._load_ok = True
    rd._map_signals()
    return rd


def truth_params(b):
    return {"a_225": 1.0, "b_elongatedRing_225": b}


def make_data(rd_truth, b_true, amp=1000.0, noise=0.03, rng=None, cal=None):
    """Observed data and errors in Emis3D's nested-list format."""
    rng = rng or np.random.default_rng(0)
    mu = np.deg2rad(INJ_DEG)
    obs, err = [], []
    for g, phi_deg in GROUPS.items():
        shape = np.array([rd_truth.data_maps["elongatedRing"]["data"][f"{g}{i}"][0] for i in range(N_CH)])
        dphi = Util_emis3D.find_dphi(np.array([np.deg2rad(phi_deg)]), mu, "elongatedRing")[0]
        clean = amp * np.exp(-b_true * dphi**2) * shape * (1.0 if cal is None else cal[g])
        sig = noise * clean
        obs.append(list(clean + rng.normal(0, sig)))
        err.append(list(sig))
    return {"observed": obs, "observed_error": err}


def prepared_candidate(rd, data_dict, fit_id=0, scale_def="exponential", vary_peak=False, method="least_squares"):
    channel_list = [[f"{g}{i}" for i in range(N_CH)] for g in GROUPS]
    data_max = max(max(v) for v in data_dict["observed"])
    rd.prepare_for_fits(channel_list, data_max=data_max)
    rd.create_parameters(enable_dphi_scaling=True, vary_peak_rad_location=vary_peak, scale_def=scale_def)
    synth = {
        "info": rd.info,
        "injectionLocation": INJ_DEG,
        "injectionLocation_rad": np.deg2rad(INJ_DEG),
        "emissionNames": rd.info["emissionNames"],
    }
    for e in rd.info["emissionNames"]:
        synth[e] = dict(rd.fitSynthetic[e])
        synth[e]["phi_array"] = rd.data["toroidalRadiatedPower"][e]["phi_array"]
        synth[e]["P_pol"] = rd.data["toroidalRadiatedPower"][e]["P_pol"]
    fit = Util_emis3D.fit_radDist(rd.fitSynthetic["params"]["params"], data_dict, synth, scale_def, 0.0, method=method)
    return (fit_id, fit, synth)


def true_tpf(b, P_pol=None):
    synth = {"injectionLocation": INJ_DEG, "injectionLocation_rad": np.deg2rad(INJ_DEG),
             "emissionNames": ["elongatedRing"],
             "elongatedRing": {"phi_array": np.linspace(0, 2 * np.pi, N_PHI, endpoint=False),
                               "P_pol": np.ones(N_PHI) if P_pol is None else P_pol, "scaleSynth": 1.0}}
    q, _ = U.derived_quantities(truth_params(b), synth, "exponential")
    return q["tpf"]


# ---------------------------------------------------------------------------
# Profile and derived quantities
# ---------------------------------------------------------------------------

def test_constant_scale_gives_tpf_one():
    synth = {"injectionLocation": INJ_DEG, "injectionLocation_rad": np.deg2rad(INJ_DEG),
             "emissionNames": ["elongatedRing"],
             "elongatedRing": {"phi_array": np.linspace(0, 2 * np.pi, N_PHI, endpoint=False),
                               "P_pol": np.full(N_PHI, 3.0), "scaleSynth": 2.0}}
    q, P = U.derived_quantities({"a_225": 1.5, "b_elongatedRing_225": 0.0}, synth, "constant")
    assert q["tpf"] == pytest.approx(1.0)
    assert q["total_power"] == pytest.approx(2 * np.pi * 3.0 * 2.0 * 1.5)   # periodic sum, no missing interval


def test_peak_is_kept_and_located_at_injection():
    tpf = true_tpf(2.0)
    assert tpf > 1.5
    rd = fake_raddist()
    synth = {"injectionLocation": INJ_DEG, "injectionLocation_rad": np.deg2rad(INJ_DEG),
             "emissionNames": ["elongatedRing"],
             "elongatedRing": {"phi_array": rd.data["toroidalRadiatedPower"]["elongatedRing"]["phi_array"],
                               "P_pol": np.ones(N_PHI), "scaleSynth": 1.0}}
    q, _ = U.derived_quantities(truth_params(2.0), synth, "exponential")
    assert abs(q["peak_phi_deg"] - INJ_DEG) <= 360 / N_PHI


def test_unknown_uncertainty_option_rejected():
    with pytest.raises(ValueError):
        U.UncertaintyConfig.from_info({"uncertainty": {"n_sample": 10}})


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def test_statistical_interval_covers_truth_at_nominal_rate():
    """Over many noisy datasets the 68% / 95% TPF intervals should contain the truth ~68% / ~95% of the time."""
    b_true = 2.0
    t = true_tpf(b_true)
    rng = np.random.default_rng(42)
    cfg = U.UncertaintyConfig(n_samples=400, seed=1)
    hits68 = hits95 = 0
    n_trials = 120
    for _ in range(n_trials):
        dd = make_data(fake_raddist(), b_true, rng=rng)
        cand = prepared_candidate(fake_raddist(), dd)
        res = U.propagate([cand], dd, "exponential", list(GROUPS), cfg)
        s = res.summary("statistical")["tpf"]
        hits68 += s["68%"][0] <= t <= s["68%"][1]
        hits95 += s["95%"][0] <= t <= s["95%"][1]
    assert 0.56 <= hits68 / n_trials <= 0.80
    assert 0.88 <= hits95 / n_trials <= 1.0


def test_model_term_weights_the_right_raddist():
    dd = make_data(fake_raddist("narrow"), 2.0, rng=np.random.default_rng(3))
    cands = [prepared_candidate(fake_raddist(k), dd, fit_id=i)
             for i, k in enumerate(["narrow", "wide", "shifted"])]
    res = U.propagate(cands, dd, "exponential", list(GROUPS), U.UncertaintyConfig(n_samples=300, min_weight=0.0))
    assert res.candidate_ids[int(np.argmax(res.weights))] == 0
    assert res.weights[0] > 0.99


def test_calibration_term_widens_the_range():
    dd = make_data(fake_raddist(), 2.0, rng=np.random.default_rng(5))
    cand = prepared_candidate(fake_raddist(), dd)
    base = U.propagate([cand], dd, "exponential", list(GROUPS), U.UncertaintyConfig(n_samples=600))
    cal = U.propagate([cand], dd, "exponential", list(GROUPS),
                      U.UncertaintyConfig(n_samples=600, calibration={g: 0.15 for g in GROUPS}, n_calibration=20))
    w_base = np.diff(base.summary("total")["tpf"]["68%"])[0]
    w_cal = np.diff(cal.summary("total")["tpf"]["68%"])[0]
    assert w_cal > 1.5 * w_base


def test_flags_parameter_at_bound():
    # Toroidally uniform data drives b to its lower bound of 0
    dd = make_data(fake_raddist(), 0.0, rng=np.random.default_rng(7))
    cand = prepared_candidate(fake_raddist(), dd)
    res = U.propagate([cand], dd, "exponential", list(GROUPS), U.UncertaintyConfig(n_samples=200))
    assert any("at lower bound" in f for f in res.flags.get(0, []))
    lo, hi = res.summary("statistical")["tpf"]["68%"]
    assert lo >= 1.0 - 1e-9                                                     # b >= 0 cannot make the TPF below 1
    best_tpf = U.derived_quantities(cand[1].params.valuesdict(), cand[2], "exponential", None)[0]["tpf"]
    assert lo - 1e-9 <= best_tpf <= hi                                          # the range contains the best fit (b on its bound)


def test_differential_evolution_fit_gets_covariance():
    dd = make_data(fake_raddist(), 2.0, rng=np.random.default_rng(9))
    cand = prepared_candidate(fake_raddist(), dd, method="differential_evolution")
    res = U.propagate([cand], dd, "exponential", list(GROUPS), U.UncertaintyConfig(n_samples=200))
    s = res.summary("statistical")["tpf"]
    assert s["68%"][1] > s["68%"][0]          # non-zero width, so a covariance was obtained


def test_fitted_peak_location_gets_a_range():
    """peak_rad_loc is derived from peak_shift; samples must recompute it, or its range is a single value."""
    dd = make_data(fake_raddist(), 2.0, rng=np.random.default_rng(11))
    cand = prepared_candidate(fake_raddist(), dd, vary_peak=True)
    assert "peak_shift" in cand[1].var_names and cand[1].params["peak_rad_loc"].expr
    res = U.propagate([cand], dd, "exponential", list(GROUPS), U.UncertaintyConfig(n_samples=400))
    lo, hi = res.summary("statistical")["peak_phi_deg"]["95%"]
    assert hi - lo > 0.1 and lo <= INJ_DEG <= hi


def test_model_weights_follow_the_ranking_score():
    dd = make_data(fake_raddist("narrow"), 2.0, rng=np.random.default_rng(3))
    cands = [prepared_candidate(fake_raddist(k), dd, fit_id=i) for i, k in enumerate(["narrow", "wide"])]
    w, best = U.candidate_weights([c[1] for c in cands])
    scores = [Util_emis3D.fit_score(c[1]) for c in cands]
    assert best == min(scores)
    assert w[1] / w[0] == pytest.approx(np.exp(-0.5 * (scores[1] - scores[0])))


# ---------------------------------------------------------------------------
# Run config and calibration
# ---------------------------------------------------------------------------

BASE = {"BOLOMETERS": {"CAM_A": {"dataFileName": "a.h5"}}, "shot": 1, "radDistDirectories_LocIndependent": ["x"]}


def test_run_config_checks_the_uncertainty_block():
    assert RunConfig.from_dict({**BASE, "uncertainty": {"n_samples": 100, "calibration": {"CAM_A": 0.1}}})
    with pytest.raises(ValueError, match="Unknown uncertainty options"):
        RunConfig.from_dict({**BASE, "uncertainty": {"n_sample": 100}})
    with pytest.raises(ValueError, match="not in BOLOMETERS"):
        RunConfig.from_dict({**BASE, "uncertainty": {"calibration": {"CAM_Z": 0.1}}})


@pytest.mark.parametrize("old, hint", [("inverse", "constant"), ("inverse square", "poisson")])
def test_renamed_error_models_explain_themselves(old, hint):
    with pytest.raises(ValueError, match=hint):
        RunConfig.from_dict({**BASE, "error_definition": old})


def test_calibration_prints_an_uncertainty_block():
    from emis3d.calibration import CalibrationResult

    f = np.array([[1.0, 0.8], [1.0, 0.84], [1.0, 0.76]])
    res = CalibrationResult(cameras=["A", "B"], times=[1, 2, 3], reference="A", amplitudes=1 / f,
                            amplitude_errors=0.01 / f, factors=f, factor_errors=0.01 * f * np.array([0.0, 1.0]),
                            best_raddist=["r"] * 3, chi2=[1.0] * 3)
    block = yaml.safe_load(res.uncertainty_block())["uncertainty"]["calibration"]
    assert block["A"] == 0.0 and block["B"] == pytest.approx(np.hypot(res.spread["B"], res.statistical_error["B"]), rel=1e-2)


def test_intervals_contain_the_best_fit_even_when_the_draws_are_skewed(monkeypatch):
    """A wide covariance or draws held on bounds can put most draws on one side of the best fit (a total-power
    range of 15 to 2e5 W around a best fit of 3.2e6 W on DIII-D); the interval must still contain it."""
    dd = make_data(fake_raddist(), 0.3, rng=np.random.default_rng(3))
    cand = prepared_candidate(fake_raddist(), dd)
    name = next(k for k in cand[1].params if k.startswith("b_"))
    best_b = cand[1].params[name].value

    def skewed(fit, n, rng):  # every draw has a much larger decay than the best fit
        base = fit.params.valuesdict()
        return [{**base, name: best_b + 5.0 + rng.uniform(0, 3)} for _ in range(n)], set()

    monkeypatch.setattr(U, "sample_parameters", skewed)
    res = U.propagate([cand], dd, "exponential", list(GROUPS), U.UncertaintyConfig(n_samples=100))
    best_q = res.best_values
    for q in ("tpf", "total_power"):
        for which in ("total", "statistical"):
            lo, hi = res.summary(which)[q]["68%"]
            assert lo - 1e-9 <= best_q[q] <= hi + 1e-9, (q, which, lo, hi, best_q[q])
    band = res.profile_band()
    assert np.all(band["68%"][0] <= res.best_profile + 1e-9) and np.all(band["68%"][1] >= res.best_profile - 1e-9)
