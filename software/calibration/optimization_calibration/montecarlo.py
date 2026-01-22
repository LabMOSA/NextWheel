from __future__ import annotations
from typing import Sequence
import numpy as np

from software.calibration.optimization_calibration.features import build_XY
from software.calibration.optimization_calibration.types import MonteCarloConfig, MonteCarloResult, BuildXYFn
from software.calibration.optimization_calibration.types import FitResult, FitConfig
from software.calibration.optimization_calibration.fit import fit


def _fit_from_trials(trials, acc_bias, base, builder: BuildXYFn, fit_configuration: FitConfig):
    X, Y_scaled, y_scale = build_XY(trials, acc_bias, base)
    fr = fit(X, Y_scaled, fit_configuration)

    A_true = (fr.A.T * y_scale).T
    return A_true


def bootstrap_A(
    trials: Sequence[dict],
    acc_bias,
    base,
    builder: BuildXYFn,
    fit_configuration: FitConfig,
    montecarlo_configuration: MonteCarloConfig,
) -> MonteCarloResult:
    """
    Bootstrap
    """
    # Create the false stochastic
    rng = np.random.default_rng(montecarlo_configuration.seed)
    N = len(trials)
    k = max(1, int(np.ceil(montecarlo_configuration.frac * N)))

    As = []
    for _ in range(montecarlo_configuration.n_draws):
        idx = rng.choice(N, size=k, replace=True)
        subset = [trials[i] for i in idx]
        As.append(_fit_from_trials(subset, acc_bias, base, builder, fit_configuration))

    As = np.stack(As, axis=0)
    return MonteCarloResult(A_mean=As.mean(axis=0), A_std=As.std(axis=0), A_samples=As)


def subsample_A(trials: Sequence[dict], acc_bias, base, builder: BuildXYFn,
                fit_cfg: FitConfig, mc_cfg: MonteCarloConfig) -> MonteCarloResult:
    rng = np.random.default_rng(mc_cfg.seed)
    N = len(trials)
    k = max(1, int(np.ceil(mc_cfg.frac * N)))

    As = []
    for _ in range(mc_cfg.n_draws):
        idx = rng.choice(N, size=k, replace=False)
        subset = [trials[i] for i in idx]
        As.append(_fit_from_trials(subset, acc_bias, base, builder, fit_cfg))

    As = np.stack(As, axis=0)
    return MonteCarloResult(A_mean=As.mean(axis=0), A_std=As.std(axis=0), A_samples=As)
