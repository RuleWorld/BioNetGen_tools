# Experimental scripts

These one-off diagnostics were written while debugging the parameter estimation tutorial in
[RuleWorld/SIAMLS26](https://github.com/RuleWorld/SIAMLS26/tree/main/notebooks). No notebook uses them, and they
are not part of the `bionetgen_tools` package or its tests. They are kept for reference.

| Script | What it does |
|---|---|
| `diagnose_nllh.py` | Splits the NLLH into residual, sigma and constant terms |
| `diagnose_sigma_issue.py` | Compares residuals at the estimated and true parameters to explain a poorly estimated noise parameter |
| `identifiability_helper_cell.py` | A notebook cell that runs the identifiability diagnosis from `bionetgen_tools.identifiability` |

They take a `ParameterEstimator` (and in some cases a `BNGLSimulator`) from `bionetgen_tools`, so they work with the
bngsim version, but they have not been tested since the port.
