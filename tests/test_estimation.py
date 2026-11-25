import numpy as np
import pandas as pd
import pytest

from bionetgen_tools import ParameterEstimator

from conftest import FCERI

TRUE = {"kp1": 1.32845238e-7, "kp2": 0.25, "kpL": 5e-2, "kmL": 20, "pLb": 30, "dm": 20}


@pytest.fixture(scope="module")
def problem(fceri):
    conds = pd.DataFrame([
        {"conditionId": "LigHigh", "conditionName": "High", "Lig_tot": 6e3},
        {"conditionId": "LigLow", "conditionName": "Low", "Lig_tot": 1e3},
    ])
    times = [0, 30, 60, 120, 300, 600, 1200]
    rng = np.random.default_rng(0)
    fceri.reset_all()
    fceri.set_parameters(TRUE)
    rows = []
    for _, c in conds.iterrows():
        r = fceri.simulate(sample_times=times, reference_values={"Lig_tot": c.Lig_tot})
        for i, t in enumerate(r["time"]):
            for o in ["BetaP2", "LynBoundU"]:
                v = r[o][i]
                rows.append(dict(observableId=o, simulationConditionId=c.conditionId,
                                 measurement=v + rng.normal(0, 0.1 * abs(v) + 0.05), time=t,
                                 observableParameters=np.nan))
    fceri.reset_all()
    obs = pd.DataFrame([
        dict(observableId="BetaP2", observableFormula="scale_B * BetaP2", noiseFormula="sigma_BetaP2"),
        dict(observableId="LynBoundU", observableFormula="LynBoundU", noiseFormula="sigma_LynBoundU"),
    ])
    params = [dict(parameterId=p, nominalValue=v, lowerBound=v / 100, upperBound=v * 100,
                   parameterScale="log10", estimate=int(p in ["kp1", "pLb", "dm", "kpL"]))
              for p, v in TRUE.items()]
    params += [
        dict(parameterId="scale_B", nominalValue=1.0, lowerBound=0.1, upperBound=10, parameterScale="log10", estimate=1),
        dict(parameterId="sigma_BetaP2", nominalValue=0.5, lowerBound=0.01, upperBound=10, parameterScale="log10", estimate=1),
        dict(parameterId="sigma_LynBoundU", nominalValue=0.5, lowerBound=0.01, upperBound=10, parameterScale="lin", estimate=1),
    ]
    est = ParameterEstimator(str(FCERI), conds, pd.DataFrame(rows), obs, pd.DataFrame(params))
    x_true = est.params_to_vector({**TRUE, "scale_B": 1.0, "sigma_BetaP2": 0.5, "sigma_LynBoundU": 0.5})
    return est, x_true


def test_gradient_matches_finite_differences(problem):
    est, x_true = problem
    x = x_true + 0.05
    f, g = est.compute_nllh_and_gradient(x)
    assert f == pytest.approx(est.compute_nllh(x), rel=1e-7)
    h = 1e-5
    for i, name in enumerate(est.estimable_params):
        e = np.zeros_like(x)
        e[i] = h
        fd = (est.compute_nllh(x + e) - est.compute_nllh(x - e)) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=1e-3, abs=1e-3), name


def test_fim_is_positive_semidefinite(problem):
    est, x_true = problem
    fim, names, idx = est.compute_fim(x_true, exclude_noise=False)
    assert names == est.estimable_params
    np.testing.assert_allclose(fim, fim.T)
    assert np.linalg.eigvalsh(fim).min() > -1e-8 * np.abs(fim).max()
    fim_model, names_model, _ = est.compute_fim(x_true)
    assert not any(n.startswith("sigma_") for n in names_model)


def test_multistart_recovers_parameters(problem):
    est, x_true = problem
    results = est.fit_multistart(n_starts=4, seed=1, verbose=False)
    best = results[0]
    # The fit should be as good as the true parameters. (Parameter values are not
    # checked: scale_B and pLb are not separately identifiable from these data.)
    assert best["nllh"] <= est.compute_nllh(x_true) + 1.0


def test_sem_overrides_noise_parameter(problem):
    """With SD;N in observableParameters, sigma_* does not enter the likelihood."""
    est, x_true = problem
    m = est.measurements_df.copy()
    m["observableParameters"] = "0.5;3"
    est2 = ParameterEstimator(est.bngl_file, est.conditions_df, m, est.observables_df, est.parameters_df)
    _, g = est2.compute_nllh_and_gradient(x_true)
    for name in ["sigma_BetaP2", "sigma_LynBoundU"]:
        assert g[est2.estimable_params.index(name)] == 0
