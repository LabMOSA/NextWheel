from dataclasses import dataclass
from typing import Optional, Callable, Sequence, Any
import numpy as np

BuildXYFn = Callable[
    [Sequence[dict], np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]


@dataclass(frozen=True)
class NormalizeConfig:
    enabled: bool = True
    method: str = "std"
    epsilon: float = 1e-12


@dataclass
class MetricResult:
    RMSE_per_axis: np.ndarray
    RMSE_total: float
    R2_per_axis: np.ndarray
    R2_total: float
    n: int


@dataclass
class InfluenceConfig:
    epsilon: float = 1e-12


@dataclass
class InfluenceResult:
    leverage: np.ndarray
    residuals: np.ndarray
    loo_residuals: np.ndarray
    delta_A_Frobenius: np.ndarray
    cook_per_axis: np.ndarray
    cook_combined: np.ndarray


@dataclass(frozen=True)
class MonteCarloConfig:
    n_draws: int = 300
    frac: float = 0.8
    seed: int = 0
    # replace: bool = True


@dataclass
class MonteCarloResult:
    A_mean: np.ndarray
    A_std: np.ndarray
    A_samples: np.ndarray


@dataclass(frozen=True)
class FitConfig:
    method: str = "ols"
    intercept: bool = False
    alpha_ridge: float = 1e-6


@dataclass
class FitResult:
    A: np.ndarray
    b0: Optional[np.ndarray]
    B: np.ndarray

@dataclass
class ReportRow:
    idx: int
    score: float
    deltaA_frob: float
    cook: float
    loo_rmse: float
    leverage: float
    meta: dict[str, Any]