import numpy as np
import pandas as pd
import pytest

from bionetgen_tools import BNGLSimulator
from bionetgen_tools._bng import model_block_only

from conftest import FCERI, ROOT


def test_ode_matches_roadrunner(fceri):
    """bngsim reproduces the libroadrunner trajectory the original notebooks used."""
    ref = pd.read_csv(ROOT / "tests" / "data" / "fceri_roadrunner_ode.csv", comment="#")
    fceri.reset_all()
    res = fceri.simulate(t_end=1200, n_steps=200)
    np.testing.assert_allclose(res["time"], ref["time"])
    for obs in fceri.observables:
        np.testing.assert_allclose(res[obs], ref[obs], rtol=1e-6, atol=1e-6, err_msg=obs)


def test_sample_times_include_zero(fceri):
    fceri.reset_all()
    res = fceri.simulate(sample_times=[30, 60, 1200])
    assert list(res["time"]) == [0, 30, 60, 1200]
    grid = fceri.simulate(t_end=1200, n_steps=40)  # every 30 s
    np.testing.assert_allclose(res["BetaP2"][1], grid["BetaP2"][1], rtol=1e-6)


def test_reference_values_recompute_seed_species(fceri):
    fceri.reset_all()
    low = fceri.simulate(t_end=600, n_steps=10, reference_values={"Lig_tot": 1000})
    high = fceri.simulate(t_end=600, n_steps=10, reference_values={"Lig_tot": 6000})
    assert high["BetaP2"][-1] > low["BetaP2"][-1]
    fceri.reset_all()
    assert fceri.get_parameter("Lig_tot") == 6000


@pytest.mark.parametrize("p", ["Lig_tot", "kp1", "pLb", "dm"])
def test_analytic_sensitivities_match_central_differences(fceri, p):
    sens = fceri.local_sensitivity_analysis(observable="BetaP1", parameters=[p], t_end=600, n_steps=20)
    h = 1e-4
    fceri.reset_all()
    v = fceri.get_parameter(p)
    fceri.set_parameter(p, v * (1 + h))
    up = fceri.simulate(600, 20)["BetaP1"][-1]
    fceri.set_parameter(p, v * (1 - h))
    down = fceri.simulate(600, 20)["BetaP1"][-1]
    fceri.reset_all()
    base = fceri.simulate(600, 20)["BetaP1"][-1]
    fd = (up - down) / (2 * h) / base
    assert sens[p][-1] == pytest.approx(fd, rel=1e-4)


def test_local_sensitivity_shapes(fceri):
    nested = fceri.local_sensitivity_analysis(observables=["BetaP1", "LynFree"], parameters=["kp1", "dm"], n_steps=50)
    assert set(nested) == {"BetaP1", "LynFree"}
    assert nested["BetaP1"]["kp1"].shape == (50,)
    assert fceri.last_time_points.shape == (50,)


def test_global_sensitivity_runs(fceri):
    coeffs, r2, outputs, log_samples = fceri.global_sensitivity_analysis(
        "BetaP1", parameters=["kp1", "pLb", "dm"], n_samples=50, oom_range=0.5, t_end=300, n_steps=10
    )
    assert np.isfinite(outputs).all()
    assert coeffs["dm"] < 0 < coeffs["pLb"]
    assert 0 < r2 <= 1


def test_ssa_seed_reproduces(fceri):
    a = fceri.simulate(t_end=300, n_steps=30, method="ssa", seed=5)
    b = fceri.simulate(t_end=300, n_steps=30, method="ssa", seed=5)
    np.testing.assert_array_equal(a["BetaP1"], b["BetaP1"])
    assert a.seed == 5


def test_replicates(ras):
    ens = ras.run_replicates(20, t_end=500, n_steps=50, method="ssa", seed=1)
    assert ens.n_replicates == 20
    assert ens["Ras_act"].shape == (20, 51)
    again = ras.simulate(t_end=500, n_steps=50, method="ssa", seed=ens.seeds[4])
    np.testing.assert_array_equal(again["Ras_act"], ens["Ras_act"][4])


def test_nfsim_runs(fceri):
    ens = fceri.run_replicates(2, t_end=60, n_steps=6, method="nf", seed=3)
    assert ens["RecDimer1"].shape == (2, 7)
    assert ens["RecDimer1"][:, -1].min() > 0


def test_write_bngl_round_trip(tmp_path):
    sim = BNGLSimulator(FCERI)
    sim.set_parameter("kp2", 0.123)
    out = tmp_path / "fit.bngl"
    sim.write_bngl(str(out), comment="test")
    text = out.read_text()
    assert "simulate(" in text  # actions are kept
    assert BNGLSimulator(out).get_parameter("kp2") == pytest.approx(0.123)


def test_model_block_only_strips_actions():
    text = FCERI.read_text()
    stripped = model_block_only(text)
    assert "generate_network" not in stripped and "simulate(" not in stripped
    assert "begin reaction rules" in stripped
