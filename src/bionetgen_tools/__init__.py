"""Tools for common modeling tasks with BioNetGen models, using bngsim.

Modules:
- simulator: BNGLSimulator (ODE and stochastic simulation, local and global
  sensitivity analysis)
- stochastic: Ensemble of replicate trajectories, with plots
- estimation: ParameterEstimator (PEtab likelihood, exact gradient, FIM,
  multistart fitting)
- fim: Fisher information analysis and parameter uncertainties
- identifiability: sensitivity, correlation and which parameters to fix
- diagnostics: checks for fixed-parameter configuration mistakes
"""

from .estimation import ParameterEstimator
from .simulator import BNGLSimulator, SimResult
from .stochastic import Ensemble, plot_distribution, plot_ensemble

__version__ = "0.1.0"

__all__ = [
    "BNGLSimulator",
    "SimResult",
    "ParameterEstimator",
    "Ensemble",
    "plot_ensemble",
    "plot_distribution",
]
