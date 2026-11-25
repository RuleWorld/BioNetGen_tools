# BioNetGen_tools

[![CI](https://github.com/RuleWorld/BioNetGen_tools/actions/workflows/ci.yml/badge.svg)](https://github.com/RuleWorld/BioNetGen_tools/actions/workflows/ci.yml)

Python tools and Jupyter notebooks for common modeling tasks with BioNetGen models: simulation, local and global
parameter sensitivity analysis, parameter estimation with identifiability analysis, and stochastic simulation. All
simulations run with [bngsim](https://github.com/lanl/bngsim), which simulates BioNetGen language (BNGL), SBML and
Antimony models by ODE integration, Gillespie's stochastic simulation algorithm (SSA), the partial-scaling
approximation (PSA), and network-free simulation with NFsim and RuleMonkey.

The notebooks began as material for the minitutorial on rule-based modeling at the SIAM Conference on the Life
Sciences (LS26), in [RuleWorld/SIAMLS26](https://github.com/RuleWorld/SIAMLS26/tree/main/notebooks), where they used
libroadrunner. This repository ports them to bngsim and collects the shared code in an installable package,
`bionetgen_tools`.

## Installation

```bash
git clone https://github.com/RuleWorld/BioNetGen_tools.git
cd BioNetGen_tools
pip install -e ".[notebooks]"
```

This installs bngsim with its `bngl` extra, which supplies BNG2.pl, the BioNetGen program that generates a reaction
network from a BNGL file. BNG2.pl is a Perl script, so `perl` must be on the path (macOS and most Linux distributions
include it). To use a BioNetGen installation you already have, set `BNGPATH` to its directory. Python 3.10 or later is
required.

## Notebooks

| Notebook | Contents | SIAMLS26 original |
|---|---|---|
| [`01_simulation_basics`](notebooks/01_simulation_basics.ipynb) | Using bngsim directly: one simulation, comparisons across parameter values, a dose scan, local sensitivities from forward sensitivity analysis, global sensitivity by Latin hypercube sampling, one SSA run | `run_bngl.ipynb` |
| [`02_sensitivity_analysis`](notebooks/02_sensitivity_analysis.ipynb) | The same analyses through `BNGLSimulator`, with a check of analytic against finite-difference sensitivities | `bngl_sensitivity_analysis.ipynb` |
| [`03_parameter_estimation`](notebooks/03_parameter_estimation.ipynb) | Synthetic data, PEtab tables, fitting by differential evolution and by gradient-based multistart, identifiability diagnosis, Fisher information analysis | `parameter_estimation_demo.ipynb` |
| [`04_stochastic_simulation`](notebooks/04_stochastic_simulation.ipynb) | SSA trajectories and distributions for a model of quantized Ras activation, replicate ensembles compared with the ODE solution, NFsim compared with SSA | `RasModelAnalyzer.ipynb` |

The notebooks read models from [`models/`](models): `fceri_starter.bngl`, an FcεRI signaling model with 11 parameters
and 6 observables, and `RasModel.bngl`, a minimal model of processive Ras activation by SOS in membrane corrals (Yonosh
and Faeder, 2025). Notebook 03 writes its PEtab files and results to `notebooks/petab_files/` and
`notebooks/estimation_results/`. On the Apple-silicon Mac used to test them, notebooks 01 and 02 run in under 10 s each, 03 in about 45 s and 04
in about 35 s.

## The package

```python
from bionetgen_tools import BNGLSimulator, ParameterEstimator

sim = BNGLSimulator("models/fceri_starter.bngl")
res = sim.simulate(t_end=1200, n_steps=200)                  # res["time"], res["BetaP2"]
sens = sim.local_sensitivity_analysis(observable="BetaP1")   # exact, from forward sensitivities
ens = sim.run_replicates(100, t_end=1200, method="ssa", seed=1)
```

| Module | Contents |
|---|---|
| `simulator` | `BNGLSimulator`: ODE and stochastic simulation (`method="ode"`, `"ssa"`, `"psa"`, `"nf"`, `"rm"`), local and global sensitivity analysis, writing a BNGL file with fitted parameter values |
| `stochastic` | `Ensemble` of replicate trajectories (means, quantiles, Fano factors, values at a time) and plots of ensembles and distributions |
| `estimation` | `ParameterEstimator`: the negative log-likelihood (NLLH) of a PEtab problem, its exact gradient, the Fisher information matrix (FIM), and multistart fitting |
| `fim` | FIM analysis: eigenvalues, parameter uncertainties from the Cramér–Rao bound, correlations |
| `identifiability` | Sensitivity and correlation analysis and a recommendation of which parameters to fix |
| `diagnostics` | Checks that fixed parameters have the values intended |

The [`docs/`](docs) folder has two guides from the original tutorial, on choosing which parameters to fix when not
all are identifiable, and on the identifiability diagnostics. One-off diagnostic scripts that no notebook uses are
in [`experimental/`](experimental).

## Changes from the SIAMLS26 versions

Local sensitivities now come from bngsim's forward sensitivity analysis (CVODES), which gives the derivatives of every
observable with respect to every parameter in one run. On the FcεRI model they agree with central finite differences
to about one part in 10⁴. The one-at-a-time finite-difference method is still available as `method="fd"`.

The parameter estimator uses the same sensitivities to compute the exact gradient of the NLLH, passed through the PEtab
observable and noise formulas by the chain rule. The gradient makes gradient-based multistart optimization practical
(`ParameterEstimator.fit_multistart`). In notebook 03, ten L-BFGS-B starts reach a slightly lower NLLH than
differential evolution with about a third as many objective evaluations, although each of those evaluations also
computes sensitivities. The FIM is now computed from the
sensitivities as the sum over data points of ∇y ∇yᵀ/σ². It is positive semidefinite by construction, so it has no
negative eigenvalues, and it costs one sensitivity run per condition instead of about 2n² objective evaluations. The
finite-difference Hessian of the original is available as `analyze_fim(..., method="hessian")`.

Simulations for fitting now report the model at the measurement times themselves, instead of at the nearest point of
a fine grid. Global sensitivity samples run in parallel threads. `write_bngl` now copies the source file and changes
only the parameter values, so rules, comments and actions are kept. Stochastic simulation, replicate ensembles and
network-free simulation are new.

In the original notebook 03, each synthetic measurement recorded its noise as `SD;N` in `observableParameters`. The
estimator then used SD/√N as the noise level and never read `sigma_BetaP2`, so the fitted noise parameter was
arbitrary (the source of the large sigma errors the original guides set out to diagnose). The synthetic data now
leave that field empty and use the noise model they were generated with, `sigma_BetaP2 * BetaP2 + 0.001`, so
`sigma_BetaP2` is the relative error and is estimated (0.098 against a true 0.10).

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests check bngsim's ODE solution of the FcεRI model against the libroadrunner trajectory the original notebooks
produced (stored in `tests/data/`), analytic sensitivities and the NLLH gradient against finite differences, that SSA
runs repeat from their seeds, and that NFsim runs. GitHub Actions runs the tests on Linux and macOS with Python 3.10
and 3.13, and executes all four notebooks, on every push and pull request and once a week, so that a new bngsim
release that breaks something is noticed.

## Citation

If you use these tools, please cite BioNetGen and bngsim:

- Harris, L. A. et al. BioNetGen 2.2: advances in rule-based modeling. Bioinformatics 32, 3366–3368 (2016).
- bngsim: https://github.com/lanl/bngsim

For PEtab, cite Schmiester, L. et al. PEtab: interoperable specification of parameter estimation problems in systems
biology. PLoS Comput. Biol. 17, e1008646 (2021). The Ras model is described in Yonosh, K. A. and Faeder, J. R.
Mathematical modeling predicts quantization of Ras activation in membrane corrals. Research Square preprint (2025),
https://www.researchsquare.com/article/rs-7889340/v1.

## License

MIT. See [LICENSE](LICENSE).
