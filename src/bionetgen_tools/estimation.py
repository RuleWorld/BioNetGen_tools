"""Parameter estimation for BNGL models from PEtab-formatted data.

``ParameterEstimator`` computes the negative log-likelihood (NLLH) of a
parameter vector given PEtab conditions, measurements, observables and
parameters tables. It is the objective for an optimizer.

With bngsim it also computes the NLLH gradient and a Fisher information
matrix from forward sensitivities. These are exact derivatives (no finite
differences), passed through the PEtab observable and noise formulas by the
chain rule. The gradient makes fast gradient-based multistart fitting
possible (``fit_multistart``); differential evolution still works as before.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

from .simulator import BNGLSimulator, SimResult

_MATH = {"log": np.log, "log10": np.log10, "exp": np.exp, "sqrt": np.sqrt}


def _names_in(formula: str) -> List[str]:
    """Identifiers a formula reads, other than the math functions."""
    try:
        tree = ast.parse(str(formula), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Cannot parse formula '{formula}': {e}")
    return sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - set(_MATH))


class ParameterEstimator:
    """Negative log-likelihood, gradient and FIM for a PEtab problem.

    One simulator is used for all conditions. Each condition's parameters
    (from the conditions table) are set before its simulation.
    """

    def __init__(
        self,
        bngl_file: str,
        conditions_df: pd.DataFrame,
        measurements_df: pd.DataFrame,
        observables_df: pd.DataFrame,
        parameters_df: pd.DataFrame,
        estimable_params: Optional[List[str]] = None,
    ):
        """
        Args:
            bngl_file: path to the BNGL model
            conditions_df, measurements_df, observables_df, parameters_df:
                the PEtab tables
            estimable_params: parameters to estimate (default: every row of
                the parameters table with estimate == 1)
        """
        self.bngl_file = bngl_file
        self.conditions_df = conditions_df
        self.measurements_df = measurements_df
        self.observables_df = observables_df
        self.parameters_df = parameters_df

        if estimable_params is None:
            self.estimable_params = parameters_df[parameters_df["estimate"] == 1]["parameterId"].tolist()
        else:
            self.estimable_params = list(estimable_params)

        self.param_info = {}
        for _, row in parameters_df.iterrows():
            self.param_info[row["parameterId"]] = {
                "scale": row["parameterScale"],
                "lower": row["lowerBound"],
                "upper": row["upperBound"],
                "nominal": row["nominalValue"],
            }

        self.timepoints = np.sort(measurements_df["time"].unique())
        self.simulator: Optional[BNGLSimulator] = None
        self._obs_formulas = {
            r["observableId"]: (r["observableFormula"], r["noiseFormula"])
            for _, r in observables_df.iterrows()
        }

        print(f"Initialized estimator with {len(self.estimable_params)} estimable parameters")
        print(f"Timepoints: {self.timepoints}")

    # ------------------------------------------------------------------
    # Parameter vectors
    # ------------------------------------------------------------------

    def vector_to_params(self, param_vector: np.ndarray) -> Dict[str, float]:
        """Estimation-space vector -> natural-space dict, with fixed parameters
        at their nominal values."""
        out = {}
        for i, name in enumerate(self.estimable_params):
            v = param_vector[i]
            out[name] = 10 ** v if self.param_info[name]["scale"] == "log10" else v
        for name, info in self.param_info.items():
            out.setdefault(name, info["nominal"])
        return out

    def params_to_vector(self, param_dict: Dict[str, float]) -> np.ndarray:
        """Natural-space dict -> estimation-space vector."""
        vec = np.zeros(len(self.estimable_params))
        for i, name in enumerate(self.estimable_params):
            v = param_dict[name]
            vec[i] = np.log10(v) if self.param_info[name]["scale"] == "log10" else v
        return vec

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds in estimation space."""
        lower = np.zeros(len(self.estimable_params))
        upper = np.zeros(len(self.estimable_params))
        for i, name in enumerate(self.estimable_params):
            info = self.param_info[name]
            if info["scale"] == "log10":
                lower[i], upper[i] = np.log10(info["lower"]), np.log10(info["upper"])
            else:
                lower[i], upper[i] = info["lower"], info["upper"]
        return lower, upper

    def _dnat_dest(self, name: str, natural_value: float) -> float:
        """d(natural value)/d(estimation-space value)."""
        return natural_value * np.log(10) if self.param_info[name]["scale"] == "log10" else 1.0

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _get_simulator(self, param_dict: Dict[str, float]) -> BNGLSimulator:
        """Create the simulator on first use, then set the model parameters."""
        if self.simulator is None:
            self.simulator = BNGLSimulator(self.bngl_file)
        sim = self.simulator
        # Parameters that are not in the model (scale_, baseline_, sigma_ and
        # so on) appear only in the observable and noise formulas.
        sim.set_parameters({k: v for k, v in param_dict.items() if k in sim.model.param_names})
        return sim

    def _model_param_names(self) -> List[str]:
        if self.simulator is None:
            self.simulator = BNGLSimulator(self.bngl_file)
        return [p for p in self.estimable_params if p in self.simulator.parameters]

    def _conditions(self):
        """Yield (condition id, condition parameters, measurements) per condition."""
        for _, row in self.conditions_df.iterrows():
            cond_id = row["conditionId"]
            params = {
                c: row[c] for c in row.index
                if c not in ("conditionId", "conditionName") and pd.notna(row[c])
            }
            meas = self.measurements_df[self.measurements_df["simulationConditionId"] == cond_id]
            yield cond_id, params, meas

    # ------------------------------------------------------------------
    # Formulas
    # ------------------------------------------------------------------

    @staticmethod
    def _namespace(sim_values: Dict[str, float], param_dict: Dict[str, float]) -> dict:
        ns = dict(param_dict)
        ns.update(sim_values)
        ns.update(_MATH)
        return ns

    @staticmethod
    def _eval(formula: str, ns: dict) -> float:
        try:
            return float(np.real(eval(str(formula), {"__builtins__": {}}, ns)))
        except Exception as e:
            raise ValueError(f"Failed to evaluate formula '{formula}': {e}")

    def _eval_observable_formula(self, formula: str, sim_result, param_dict: Dict[str, float], time_idx: int) -> float:
        """Evaluate a PEtab formula at one output time of a simulation."""
        values = {k: sim_result[k][time_idx] for k in sim_result.keys() if k != "time"}
        return self._eval(formula, self._namespace(values, param_dict))

    @staticmethod
    def _formula_partials(formula: str, ns: dict, names: Sequence[str]) -> Dict[str, float]:
        """Partial derivatives of a formula with respect to the given names,
        by complex step (exact for analytic formulas), falling back to central
        differences."""
        out = {}
        for n in names:
            x = ns[n]
            try:
                h = 1e-30 * max(abs(x), 1.0)
                ns_c = dict(ns)
                ns_c[n] = complex(x, h)
                val = eval(str(formula), {"__builtins__": {}}, ns_c)
                out[n] = float(np.imag(val) / h)
            except Exception:
                h = 1e-6 * max(abs(x), 1e-12)
                ns_p, ns_m = dict(ns), dict(ns)
                ns_p[n], ns_m[n] = x + h, x - h
                out[n] = (ParameterEstimator._eval(formula, ns_p) - ParameterEstimator._eval(formula, ns_m)) / (2 * h)
        return out

    def _parse_observable_parameters(self, obs_params) -> Dict[str, float]:
        """observableParameters: empty, or "SD;N" (the SEM is SD/sqrt(N))."""
        if pd.isna(obs_params) or obs_params == "":
            return {}
        parts = str(obs_params).split(";")
        if len(parts) == 2:
            try:
                sd, n = float(parts[0]), int(parts[1])
                return {"SD": sd, "N": n, "SEM": sd / np.sqrt(n)}
            except ValueError:
                pass
        return {f"param{i}": p for i, p in enumerate(parts)}

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _evaluate(self, param_vector: np.ndarray, want_derivs: bool, verbose: bool = False):
        """Shared core: returns (nllh, rows) where each row holds the residual,
        sigma, and (if requested) gradients of prediction and sigma with
        respect to the estimable parameters in estimation space."""
        param_dict = self.vector_to_params(param_vector)
        est = self.estimable_params
        n_est = len(est)
        model_est = self._model_param_names() if want_derivs else []
        nllh, rows = 0.0, []

        for cond_id, cond_params, meas in self._conditions():
            if len(meas) == 0:
                continue
            sim = self._get_simulator(param_dict)
            sim.set_parameters(cond_params)
            sim.model.reset()
            times = np.sort(meas["time"].unique()).astype(float)
            # A parameter the condition sets does not vary with the estimate.
            sens_params = [p for p in model_est if p not in cond_params]
            try:
                raw = sim._run("ode", times[-1], 1, times, None, start="initial",
                               sensitivity_params=sens_params)
            except Exception as e:
                if verbose:
                    print(f"Simulation failed for condition {cond_id}: {e}")
                return np.inf, None
            result = SimResult(raw)
            if sens_params:
                S = np.asarray(raw.sensitivities_observables)
                obs_names = list(raw.observable_names)

            for _, m in meas.iterrows():
                obs_id = m["observableId"]
                if obs_id not in self._obs_formulas:
                    if verbose:
                        print(f"Warning: Observable {obs_id} not found in observables table")
                    return np.inf, None
                obs_formula, noise_formula = self._obs_formulas[obs_id]
                t_idx = int(np.argmin(np.abs(result["time"] - m["time"])))
                values = {k: result[k][t_idx] for k in result.observable_names}
                ns = self._namespace(values, param_dict)
                try:
                    pred = self._eval(obs_formula, ns)
                except ValueError as e:
                    if verbose:
                        print(f"Failed to evaluate observable {obs_id}: {e}")
                    return np.inf, None

                sem = self._parse_observable_parameters(m.get("observableParameters", "")).get("SEM")
                if sem is not None:
                    sigma = sem
                else:
                    try:
                        sigma = self._eval(noise_formula, ns)
                    except ValueError as e:
                        if verbose:
                            print(f"Failed to evaluate noise formula {noise_formula}: {e}")
                        return np.inf, None

                r = m["measurement"] - pred
                nllh += 0.5 * (r / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2 * np.pi)
                row = {"r": r, "sigma": sigma}

                if want_derivs:
                    gy, gs = np.zeros(n_est), np.zeros(n_est)
                    for formula, g, active in ((obs_formula, gy, True), (noise_formula, gs, sem is None)):
                        if not active:
                            continue
                        refs = [n for n in _names_in(formula) if n in ns]
                        partials = self._formula_partials(formula, ns, refs)
                        for name, d in partials.items():
                            if name in values and sens_params:
                                # model observable: chain through dO/dp
                                io = obs_names.index(name)
                                for k, p in enumerate(sens_params):
                                    g[est.index(p)] += d * S[t_idx, io, k]
                            if name in est:
                                # parameter read by the formula itself
                                g[est.index(name)] += d
                    for k, p in enumerate(est):
                        scale = self._dnat_dest(p, param_dict[p])
                        gy[k] *= scale
                        gs[k] *= scale
                    row["grad_pred"], row["grad_sigma"] = gy, gs
                rows.append(row)

        if verbose:
            print(f"NLLH = {nllh:.4f} ({len(rows)} datapoints)")
        return nllh, rows

    def compute_nllh(self, param_vector: np.ndarray, verbose: bool = False) -> float:
        """Negative log-likelihood (Gaussian noise) at a parameter vector in
        estimation space. This is the objective to minimize."""
        nllh, _ = self._evaluate(np.asarray(param_vector, dtype=float), want_derivs=False, verbose=verbose)
        return nllh

    def compute_nllh_and_gradient(self, param_vector: np.ndarray) -> Tuple[float, np.ndarray]:
        """NLLH and its exact gradient in estimation space, from forward
        sensitivities. The pair is what ``scipy.optimize.minimize(jac=True)``
        expects."""
        nllh, rows = self._evaluate(np.asarray(param_vector, dtype=float), want_derivs=True)
        n = len(self.estimable_params)
        if rows is None or not np.isfinite(nllh):
            return 1e300, np.zeros(n)
        grad = np.zeros(n)
        for row in rows:
            r, s = row["r"], row["sigma"]
            grad += -(r / s**2) * row["grad_pred"] + (1 / s - r**2 / s**3) * row["grad_sigma"]
        return nllh, grad

    def compute_fim(self, param_vector: np.ndarray, exclude_noise: bool = True) -> Tuple[np.ndarray, List[str], List[int]]:
        """Fisher information matrix in estimation space, from sensitivities.

        For Gaussian noise, FIM = sum over data of
        (1/sigma²) ∇y ∇yᵀ + (2/sigma²) ∇sigma ∇sigmaᵀ.
        It is positive semidefinite by construction, so it has no negative
        eigenvalues; zero eigenvalues mean non-identifiable directions.

        Returns:
            (fim, parameter names, indices into the estimation vector)
        """
        _, rows = self._evaluate(np.asarray(param_vector, dtype=float), want_derivs=True)
        if rows is None:
            raise RuntimeError("Simulation failed at this parameter vector")
        idx = [i for i, p in enumerate(self.estimable_params)
               if not (exclude_noise and p.startswith("sigma_"))]
        names = [self.estimable_params[i] for i in idx]
        fim = np.zeros((len(idx), len(idx)))
        for row in rows:
            s = row["sigma"]
            gy = row["grad_pred"][idx]
            gs = row["grad_sigma"][idx]
            fim += np.outer(gy, gy) / s**2 + 2 * np.outer(gs, gs) / s**2
        return fim, names, idx

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_multistart(
        self,
        n_starts: int = 10,
        seed: int = 0,
        method: str = "L-BFGS-B",
        options: Optional[dict] = None,
        verbose: bool = True,
    ) -> List[dict]:
        """Gradient-based multistart optimization.

        Starting points are a Latin hypercube over the bounds (in estimation
        space). Each start runs ``scipy.optimize.minimize`` with the exact
        gradient.

        Returns:
            one dict per start, sorted by NLLH, with the same keys as the
            trial results in the parameter estimation notebook
        """
        lower, upper = self.get_bounds()
        starts = qmc.scale(qmc.LatinHypercube(d=len(lower), seed=seed).random(n_starts), lower, upper)
        results = []
        for i, x0 in enumerate(starts):
            res = minimize(self.compute_nllh_and_gradient, x0, jac=True, method=method,
                           bounds=list(zip(lower, upper)), options=options or {"maxiter": 500})
            results.append({
                "trial_idx": i,
                "success": bool(res.success),
                "nllh": float(res.fun),
                "nfev": int(res.nfev),
                "nit": int(res.nit),
                "param_vector": res.x,
                "param_dict": self.vector_to_params(res.x),
                "message": str(res.message),
            })
            if verbose:
                print(f"  start {i+1:3d}/{n_starts}: NLLH = {res.fun:12.4f}  ({res.nit} iterations, {res.nfev} evaluations)")
        results.sort(key=lambda r: r["nllh"])
        return results

    # ------------------------------------------------------------------
    # Simulation output and plots
    # ------------------------------------------------------------------

    def simulate_conditions(self, param_vector: np.ndarray, n_steps: int = 200) -> Dict[str, SimResult]:
        """Simulate every condition on an even time grid."""
        param_dict = self.vector_to_params(param_vector)
        results = {}
        for cond_id, cond_params, _ in self._conditions():
            sim = self._get_simulator(param_dict)
            results[cond_id] = sim.simulate(
                t_end=float(self.timepoints[-1]), n_steps=n_steps,
                reference_values=cond_params, reset=True,
            )
        return results

    def plot_fit(self, param_vector: np.ndarray, figsize: Optional[Tuple[int, int]] = None,
                 save_path: Optional[str] = None):
        """Model trajectories against the data, one panel per observable and condition."""
        param_dict = self.vector_to_params(param_vector)
        results = self.simulate_conditions(param_vector, n_steps=200)
        unique_obs = self.measurements_df["observableId"].unique()
        n_cond, n_obs = len(self.conditions_df), len(unique_obs)
        figsize = figsize or (5 * n_cond, 4 * n_obs)
        fig, axes = plt.subplots(n_obs, n_cond, figsize=figsize, squeeze=False)

        for oi, obs_id in enumerate(unique_obs):
            formula, noise_formula = self._obs_formulas[obs_id]
            for ci, (_, cond_row) in enumerate(self.conditions_df.iterrows()):
                cond_id = cond_row["conditionId"]
                ax = axes[oi, ci]
                res = results[cond_id]
                curve = [self._eval_observable_formula(formula, res, param_dict, t) for t in range(len(res["time"]))]
                ax.plot(res["time"], curve, "-", linewidth=2, label="Model")

                data = self.measurements_df[
                    (self.measurements_df["simulationConditionId"] == cond_id)
                    & (self.measurements_df["observableId"] == obs_id)
                ]
                if len(data):
                    sems = [self._parse_observable_parameters(r.get("observableParameters", "")).get("SEM")
                            for _, r in data.iterrows()]
                    if any(s is not None for s in sems):
                        ax.errorbar(data["time"], data["measurement"], yerr=sems, fmt="o",
                                    markersize=6, capsize=4, label="Data ± SEM")
                    else:
                        # Error bars from the fitted noise model, at the nearest output time
                        sigmas = [
                            self._eval_observable_formula(
                                noise_formula, res, param_dict,
                                int(np.argmin(np.abs(res["time"] - t))))
                            for t in data["time"]
                        ]
                        ax.errorbar(data["time"], data["measurement"], yerr=sigmas, fmt="o",
                                    markersize=6, capsize=4, label="Data ± model σ")
                ax.set_xlabel("Time", fontweight="bold")
                ax.set_ylabel(obs_id, fontweight="bold")
                ax.set_title(f"{cond_id}", fontweight="bold")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {save_path}")
        plt.show()
