"""Simulation and sensitivity analysis for BNGL models, using bngsim.

``BNGLSimulator`` loads a BNGL file with bngsim (which generates the reaction
network with BNG2.pl), runs it deterministically (ODE) or stochastically (SSA,
PSA, NFsim, RuleMonkey), and performs local and global sensitivity analysis.

Local sensitivities come from bngsim's forward sensitivity analysis (CVODES),
which gives exact derivatives along the trajectory in one run. The
finite-difference method used in the original SIAM LS26 notebooks is kept as
``method="fd"`` for comparison.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import bngsim
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
from sklearn.linear_model import LinearRegression

from ._bng import write_bng_xml

ODE_METHODS = {"ode"}
NETWORK_FREE_METHODS = {"nf", "nfsim", "nf_reject", "rm", "rulemonkey", "nf_exact"}
STOCHASTIC_METHODS = {"ssa", "psa"} | NETWORK_FREE_METHODS


class SimResult(dict):
    """One simulation: time and observables, indexed by name.

    ``result['time']`` and ``result['BetaP2']`` are 1-D NumPy arrays.
    ``result['Time']`` also works, for code written against libroadrunner.
    The full bngsim ``Result`` (species, sensitivities, seed, solver
    statistics) is ``result.raw``.
    """

    def __init__(self, raw: "bngsim.Result"):
        super().__init__()
        self.raw = raw
        self["time"] = np.array(raw.time, dtype=float)
        for name in raw.observable_names:
            self[name] = np.array(raw.observables[name], dtype=float)

    def __getitem__(self, key):
        if key == "Time":
            key = "time"
        return super().__getitem__(key)

    @property
    def time(self) -> np.ndarray:
        return self["time"]

    @property
    def observable_names(self) -> List[str]:
        return [k for k in self.keys() if k != "time"]

    @property
    def colnames(self) -> List[str]:
        return ["time"] + self.observable_names

    @property
    def species(self) -> np.ndarray:
        return np.asarray(self.raw.species)

    @property
    def seed(self):
        return self.raw.seed

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame({k: v for k, v in self.items()})


class BNGLSimulator:
    """Simulate a BNGL model and analyze its parameter sensitivity.

    Attributes:
        model: the bngsim ``Model``
        observables: observable names
        parameters: parameters that can be set independently (bngsim's
            primary parameters, without internal names starting with '_')
    """

    def __init__(
        self,
        bngl_file: Union[str, Path],
        param_values: Optional[Dict[str, float]] = None,
        rtol: float = 1e-9,
        atol: float = 1e-12,
        max_steps: int = 20000,
    ):
        """
        Args:
            bngl_file: path to the BNGL model file
            param_values: parameter values that override those in the file.
                They become the nominal values that ``reset_all()`` restores.
            rtol, atol, max_steps: ODE solver settings. The defaults are the
                ones the original notebooks set on libroadrunner.
        """
        self.bngl_file = str(bngl_file)
        self.model = bngsim.Model.from_bngl(self.bngl_file)
        self.rtol, self.atol, self.max_steps = rtol, atol, max_steps

        self.observables = list(self.model.observable_names)
        self.parameters = [
            p for p in self.model.primary_param_names if not p.startswith("_")
        ]

        if param_values:
            for name, value in param_values.items():
                if name in self.model.param_names:
                    self.model.set_param(name, float(value))

        self._nominal = {p: self.model.get_param(p) for p in self.parameters}
        self.model.reset()
        self._sims: Dict[tuple, "bngsim.Simulator"] = {}
        self.last_time_points: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def set_parameter(self, param_name: str, value: float):
        """Set a parameter. Takes effect at the next simulation; seed species
        that depend on it are recomputed when the model is reset."""
        self.model.set_param(param_name, float(value))

    def get_parameter(self, param_name: str) -> float:
        return self.model.get_param(param_name)

    def set_parameters(self, values: Dict[str, float]):
        for name, value in values.items():
            self.set_parameter(name, value)

    @property
    def nominal_values(self) -> Dict[str, float]:
        """Parameter values that ``reset_all()`` restores."""
        return dict(self._nominal)

    def reset_all(self):
        """Restore nominal parameter values and initial conditions."""
        self.model.set_params(self._nominal)
        self.model.reset()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulator(self, method: str = "ode", sensitivity_params: Sequence[str] = ()):
        key = (method, tuple(sensitivity_params))
        if key not in self._sims:
            kwargs = {}
            if sensitivity_params:
                kwargs["sensitivity_params"] = list(sensitivity_params)
            if method in NETWORK_FREE_METHODS:
                kwargs["xml_path"] = str(write_bng_xml(self.bngl_file))
            self._sims[key] = bngsim.Simulator(self.model, method=method, **kwargs)
        return self._sims[key]

    def _ode_kwargs(self) -> dict:
        return dict(rtol=self.rtol, atol=self.atol, max_steps=self.max_steps)

    def _run(
        self,
        method: str,
        t_end: float,
        n_steps: int,
        sample_times: Optional[Sequence[float]],
        seed: Optional[int],
        start: str,
        sensitivity_params: Sequence[str] = (),
        t_start: float = 0.0,
    ):
        sim = self._simulator(method, sensitivity_params)
        kwargs = {"start": start}
        if method in ODE_METHODS:
            kwargs.update(self._ode_kwargs())
        else:
            kwargs["seed"] = seed
        if sample_times is not None:
            times = sorted(set(float(t) for t in sample_times) | {float(t_start)})
            # bngsim starts integrating at the first sample time, so t_start
            # must be included explicitly.
            return sim.run(sample_times=times, **kwargs)
        return sim.run(t_span=(float(t_start), float(t_end)), n_points=int(n_steps) + 1, **kwargs)

    def simulate(
        self,
        t_end: float = 1200,
        n_steps: int = 200,
        reset: bool = True,
        reference_values: Optional[Dict[str, float]] = None,
        method: str = "ode",
        seed: Optional[int] = None,
        sample_times: Optional[Sequence[float]] = None,
    ) -> SimResult:
        """Run one simulation with the current parameter values.

        Args:
            t_end: end time
            n_steps: number of output intervals (n_steps + 1 points, as in
                BioNetGen's simulate action)
            reset: start from the initial conditions. False continues from
                the state the last run ended in.
            reference_values: parameters to set before the run (for example,
                a ligand dose for one experimental condition). They stay set.
            method: "ode", "ssa", "psa", "nf" (NFsim) or "rm" (RuleMonkey)
            seed: random seed for stochastic methods (None draws a fresh one;
                the seed used is ``result.seed``)
            sample_times: output times, instead of an even grid. Time 0 is
                always included.

        Returns:
            SimResult with 'time' and one array per observable
        """
        if reference_values:
            self.set_parameters(reference_values)
        if reset:
            self.model.reset()
        raw = self._run(
            method, t_end, n_steps, sample_times, seed,
            start="initial" if reset else "live",
        )
        return SimResult(raw)

    def run_replicates(
        self,
        n_replicates: int,
        t_end: float = 1200,
        n_steps: int = 200,
        method: str = "ssa",
        seed: Optional[int] = None,
        reference_values: Optional[Dict[str, float]] = None,
        num_processors: Optional[int] = None,
    ):
        """Run independent stochastic trajectories from the initial conditions.

        SSA and PSA replicates run in parallel threads inside bngsim. NFsim and
        RuleMonkey replicates run one after another, each with its own seed
        derived from ``seed``.

        Returns:
            an ``Ensemble`` (see ``bionetgen_tools.stochastic``)
        """
        from .stochastic import Ensemble

        if method not in STOCHASTIC_METHODS:
            raise ValueError(f"run_replicates needs a stochastic method, not '{method}'")
        if reference_values:
            self.set_parameters(reference_values)
        self.model.reset()

        if method in NETWORK_FREE_METHODS:
            base = seed if seed is not None else int(np.random.SeedSequence().entropy % 2**31)
            seeds = [base] + [
                int(s) & 0x7FFFFFFF
                for s in np.random.SeedSequence(base).generate_state(n_replicates - 1)
            ]
            runs = []
            for s in seeds:
                self.model.reset()
                runs.append(self._run(method, t_end, n_steps, None, s, start="initial"))
        else:
            sim = self._simulator(method)
            runs = sim.run_replicates(
                n_replicates, t_span=(0.0, float(t_end)), n_points=int(n_steps) + 1,
                seed=seed, num_processors=num_processors,
            )
        self.model.reset()
        return Ensemble.from_results(runs, method=method)

    # ------------------------------------------------------------------
    # Writing the model back out
    # ------------------------------------------------------------------

    def write_bngl(self, output_file: str, comment: Optional[str] = None):
        """Write a copy of the BNGL file with the current parameter values.

        Only the values in the parameters block change; rules, comments and
        actions are kept as they are in the source file.
        """
        text = Path(self.bngl_file).read_text()
        current = {p: self.get_parameter(p) for p in self.parameters}
        out, in_params = [], False
        line_re = re.compile(r"^(\s*(?:\d+\s+)?)([A-Za-z_]\w*)(\s*=?\s*)(\S+)(.*)$")
        for line in text.splitlines():
            stripped = line.strip().lower()
            if stripped.startswith("begin parameters"):
                in_params = True
            elif stripped.startswith("end parameters"):
                in_params = False
            elif in_params and not stripped.startswith("#"):
                m = line_re.match(line)
                if m and m.group(2) in current:
                    lead, name, sep, _old, rest = m.groups()
                    line = f"{lead}{name}{sep}{current[name]:.10g}{rest}"
            out.append(line)
        header = "# BNGL model written by bionetgen_tools.BNGLSimulator\n"
        if comment:
            header += f"# {comment}\n"
        Path(output_file).write_text(header + "\n".join(out) + "\n")
        print(f"BNGL file written to: {output_file}")

    # ------------------------------------------------------------------
    # Local sensitivity analysis
    # ------------------------------------------------------------------

    def local_sensitivity_analysis(
        self,
        observables: Optional[Union[str, List[str]]] = None,
        parameters: Optional[List[str]] = None,
        reference_values: Optional[Dict[str, float]] = None,
        t_end: float = 600,
        n_steps: int = 200,
        perturbation: float = 1e-4,
        method: str = "analytic",
        observable: Optional[str] = None,
    ) -> Dict:
        """Normalized local sensitivities, (dO/O)/(dp/p), along a trajectory.

        Args:
            observables: observables to analyze (None = all)
            parameters: parameters (None = all)
            reference_values: parameter values for the reference point, for
                example a ligand dose. Others take their nominal values.
            t_end, n_steps: time course
            perturbation: relative step for method="fd"
            method: "analytic" uses bngsim forward sensitivities (exact, one
                run). "fd" perturbs one parameter at a time (forward
                differences), as the original notebooks did.
            observable: a single observable. When given, the result is
                ``{parameter: array}`` instead of the nested form.

        Returns:
            ``{observable: {parameter: sensitivity_array}}`` (or
            ``{parameter: array}`` when ``observable`` is given). Arrays skip
            t = 0; their times are ``self.last_time_points``. Where the
            observable is zero the absolute sensitivity dO/dp is returned
            instead, as in the original code.
        """
        single = observable is not None
        if single:
            observables = [observable]
        elif observables is None:
            observables = list(self.observables)
        elif isinstance(observables, str):
            observables = [observables]
        for obs in observables:
            if obs not in self.observables:
                raise ValueError(f"Observable '{obs}' not found in model")

        if parameters is None:
            parameters = list(self.parameters)
        params = []
        for p in parameters:
            if p in self.parameters:
                params.append(p)
            else:
                print(f"Warning: Parameter '{p}' not found, skipping")

        self.reset_all()
        if reference_values:
            for name, value in reference_values.items():
                if name in self.parameters:
                    self.set_parameter(name, value)
                else:
                    print(f"Warning: Reference parameter '{name}' not found in model")

        if method == "analytic":
            sens = self._analytic_sensitivities(observables, params, t_end, n_steps)
        elif method == "fd":
            sens = self._fd_sensitivities(observables, params, reference_values, t_end, n_steps, perturbation)
        else:
            raise ValueError("method must be 'analytic' or 'fd'")

        self.reset_all()
        return sens[observables[0]] if single else sens

    def _analytic_sensitivities(self, observables, params, t_end, n_steps):
        raw = self._run("ode", t_end, n_steps, None, None, start="initial", sensitivity_params=params)
        names = list(raw.observable_names)
        O = np.column_stack([np.asarray(raw.observables[n]) for n in names])
        S = np.asarray(raw.sensitivities_observables)  # (n_times, n_obs, n_params)
        cols = list(raw.sensitivity_params)
        self.last_time_points = np.asarray(raw.time)[1:]

        out = {obs: {} for obs in observables}
        for obs in observables:
            io = names.index(obs)
            O_nom = O[1:, io]
            for p in params:
                dOdp = S[1:, io, cols.index(p)]
                pval = self.get_parameter(p)
                with np.errstate(divide="ignore", invalid="ignore"):
                    normalized = dOdp * pval / O_nom
                out[obs][p] = np.where(np.isfinite(normalized), normalized, dOdp)
        return out

    def _fd_sensitivities(self, observables, params, reference_values, t_end, n_steps, perturbation):
        res_nom = self.simulate(t_end, n_steps)
        O_nom = {obs: res_nom[obs][1:] for obs in observables}
        self.last_time_points = res_nom["time"][1:]
        out = {obs: {} for obs in observables}
        for p in params:
            self.reset_all()
            if reference_values:
                for name, value in reference_values.items():
                    if name in self.parameters:
                        self.set_parameter(name, value)
            orig = self.get_parameter(p)
            self.set_parameter(p, (1 + perturbation) * orig)
            res_pert = self.simulate(t_end, n_steps)
            for obs in observables:
                O_pert = res_pert[obs][1:]
                with np.errstate(divide="ignore", invalid="ignore"):
                    normalized = ((O_pert - O_nom[obs]) / O_nom[obs]) / perturbation
                    absolute = (O_pert - O_nom[obs]) / (perturbation * orig)
                out[obs][p] = np.where(np.isfinite(normalized), normalized, absolute)
        return out

    @staticmethod
    def _flat(sensitivities: Dict, observable: str) -> Dict[str, np.ndarray]:
        """Accept either the nested or the single-observable form."""
        if observable in sensitivities and isinstance(sensitivities[observable], dict):
            return sensitivities[observable]
        return sensitivities

    def plot_local_sensitivity(
        self,
        sensitivities: Dict,
        observable: str,
        time_index: int = -1,
        top_n: Optional[int] = None,
        thresh: float = 0.0,
        figsize: Tuple[int, int] = (10, 6),
    ):
        """Bar chart of local sensitivities at one time point."""
        sens = self._flat(sensitivities, observable)
        params_values = [(p, v[time_index]) for p, v in sens.items() if abs(v[time_index]) > thresh]
        if not params_values:
            print(f"No parameters with |sensitivity| > {thresh}")
            return
        params_values.sort(key=lambda x: abs(x[1]), reverse=True)
        if top_n is not None:
            params_values = params_values[:top_n]
        params = [p for p, _ in params_values]
        values = [v for _, v in params_values]
        colors = ["tab:blue" if v < 0 else "tab:red" for v in values]

        plt.figure(figsize=figsize)
        plt.bar(params, values, color=colors)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Normalized Sensitivity")
        title = f"Local Sensitivity of {observable} at t={self.last_time_points[time_index]:.1f}s"
        if thresh > 0:
            title += f" (|S| > {thresh})"
        if top_n is not None:
            title += f" (top {len(params)})"
        plt.title(title)
        plt.axhline(0, ls="--", color="k", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_sensitivity_trajectories(
        self,
        sensitivities: Dict,
        observable: str,
        top_n: Optional[int] = 10,
        thresh: float = 0.0,
        figsize: Tuple[int, int] = (12, 6),
    ):
        """Sensitivity of one observable over time, for the most sensitive parameters."""
        sens = self._flat(sensitivities, observable)
        params = [p for p in sens if abs(sens[p][-1]) > thresh]
        if not params:
            print(f"No parameters with |sensitivity| > {thresh}")
            return
        params.sort(key=lambda p: abs(sens[p][-1]), reverse=True)
        if top_n is not None:
            params = params[:top_n]

        plt.figure(figsize=figsize)
        for p in params:
            plt.plot(self.last_time_points, sens[p], label=p)
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Sensitivity")
        title = f"Sensitivity Trajectories for {observable}"
        if thresh > 0:
            title += f" (|S| > {thresh})"
        plt.title(title)
        plt.axhline(0, ls="--", color="k", alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Global sensitivity analysis
    # ------------------------------------------------------------------

    def global_sensitivity_analysis(
        self,
        observable: str,
        parameters: Optional[List[str]] = None,
        t_end: float = 600,
        n_steps: int = 200,
        n_samples: int = 1000,
        oom_range: float = 2.0,
        seed: int = 42,
        num_processors: Optional[int] = None,
    ) -> Tuple[Dict[str, float], float, np.ndarray, np.ndarray]:
        """Global sensitivity by Latin hypercube sampling and log-log regression.

        Each parameter is sampled over ±``oom_range`` orders of magnitude
        around its current value, the observable at ``t_end`` is recorded,
        and log10(output) is regressed on log10(parameters). The regression
        coefficients are the sensitivity coefficients.

        Samples run in parallel threads with bngsim's ``run_batch``. If any
        sample fails, the batch is rerun one sample at a time and failed
        samples are recorded as NaN.

        Returns:
            (sensitivity_coeffs, r2_score, outputs, log_samples)
        """
        if parameters is None:
            parameters = list(self.parameters)
        if observable not in self.observables:
            raise ValueError(f"Observable '{observable}' not found in model")
        n_params = len(parameters)

        lower, upper = [], []
        for p in parameters:
            nominal = self.get_parameter(p)
            if nominal > 0:
                lower.append(np.log10(nominal) - oom_range)
                upper.append(np.log10(nominal) + oom_range)
            else:
                lower.append(-6)
                upper.append(-2)

        print(f"Sampling {n_params} parameters using Latin Hypercube")
        print(f"Parameters: {parameters}")
        sampler = qmc.LatinHypercube(d=n_params, seed=seed)
        unit = sampler.random(n=n_samples)
        log_samples = qmc.scale(unit, l_bounds=np.array(lower), u_bounds=np.array(upper))
        param_samples = 10 ** log_samples
        print(f"Generated {n_samples} samples")
        print(f"Sample quality (discrepancy): {qmc.discrepancy(unit):.4f}")

        point_sets = [dict(zip(parameters, map(float, row))) for row in param_samples]
        print("\nRunning simulations...")
        outputs = self._batch_final_values(observable, point_sets, t_end, n_steps, num_processors)

        failed = np.isnan(outputs)
        n_failed = int(failed.sum())
        n_success = n_samples - n_failed
        print("\nSimulation complete!")
        print(f"  Successful: {n_success}/{n_samples} ({100*n_success/n_samples:.1f}%)")
        print(f"  Failed: {n_failed}/{n_samples} ({100*n_failed/n_samples:.1f}%)")

        valid = ~failed
        print(f"\nOutput range (valid): [{outputs[valid].min():.2e}, {outputs[valid].max():.2e}]")
        X = log_samples[valid]
        y = np.log10(outputs[valid] + 1e-10)
        reg = LinearRegression().fit(X, y)
        coeffs = dict(zip(parameters, reg.coef_))
        r2 = reg.score(X, y)

        print(f"\nR² score: {r2:.4f}")
        print("\nGlobal sensitivity coefficients:")
        for name, c in coeffs.items():
            print(f"  {name:12s}: {c:+.4f}")

        self.reset_all()
        return coeffs, r2, outputs, log_samples

    def _batch_final_values(self, observable, point_sets, t_end, n_steps, num_processors):
        self.reset_all()
        sim = self._simulator("ode")
        span = dict(t_span=(0.0, float(t_end)), n_points=int(n_steps) + 1)
        try:
            rows = sim.run_batch(params=point_sets, num_processors=num_processors, **span, **self._ode_kwargs())
            out = np.array([float(r.observables[observable][-1]) for r in rows])
            out[~np.isfinite(out)] = np.nan
            return out
        except bngsim.SimulationError:
            print("  A sample failed in the parallel batch; rerunning one at a time")

        out = np.full(len(point_sets), np.nan)
        n_failed = 0
        for i, point in enumerate(point_sets):
            try:
                self.reset_all()
                self.set_parameters(point)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = self.simulate(t_end, n_steps)
                val = res[observable][-1]
                if np.isfinite(val):
                    out[i] = val
                else:
                    n_failed += 1
            except Exception as e:
                n_failed += 1
                if n_failed <= 5:
                    print(f"  Simulation {i} failed: {type(e).__name__}")
            if (i + 1) % 100 == 0:
                print(f"  Completed {i+1}/{len(point_sets)} simulations ({n_failed} failed so far)")
        return out

    def plot_global_sensitivity(
        self,
        sensitivity_coeffs: Dict[str, float],
        r2_score: float,
        observable: str,
        outputs: np.ndarray,
        log_samples: np.ndarray,
        t_end: float,
        thresh: float = 0.0,
        figsize: Tuple[int, int] = (14, 5),
    ):
        """Bar chart of global sensitivity coefficients, and predicted vs actual output."""
        params = [p for p, c in sensitivity_coeffs.items() if abs(c) > thresh]
        coeffs = [sensitivity_coeffs[p] for p in params]
        if not params:
            print(f"No parameters with |sensitivity| > {thresh}")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.bar(params, coeffs, color=["tab:blue" if c < 0 else "tab:red" for c in coeffs])
        ax1.set_xticks(range(len(params)))
        ax1.set_xticklabels(params, rotation=45, ha="right")
        ax1.set_ylabel("Sensitivity Coefficient")
        title = f"Global Sensitivity of {observable} at t={t_end}s"
        if thresh > 0:
            title += f"\n(|S| > {thresh})"
        ax1.set_title(title)
        ax1.axhline(0, ls="--", color="k", alpha=0.3)

        valid = ~np.isnan(outputs)
        X = log_samples[valid]
        y = np.log10(outputs[valid] + 1e-10)
        y_pred = LinearRegression().fit(X, y).predict(X)
        ax2.scatter(y, y_pred, alpha=0.5, s=30)
        lo, hi = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        ax2.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect prediction")
        ax2.set_xlabel(f"Actual log₁₀({observable})")
        ax2.set_ylabel(f"Predicted log₁₀({observable})")
        ax2.set_title(f"Prediction Accuracy (R²={r2_score:.3f})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
