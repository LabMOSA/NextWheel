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
    deltaA_Frobenius: float
    cook: float
    loo_rmse: float
    leverage: float
    meta: dict[str, Any]

@dataclass
class ProtocolEval:
    name: str
    n_total: int
    n_train: int
    n_test: int

    rmse_total: float
    r2_total: float
    rmse_per_axis: list[float]
    r2_per_axis: list[float]

    # Stability (bootstrap)
    A_std_mean: float
    A_std_max: float
    A_cv_mean: float
    A_cv_max: float

    # Influence summary (optional)
    influence_top_score: float | None
    influence_top_idx: int | None
    influence_top_file: str | None

    # Debug/info
    notes: str = ""

@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    masses: tuple[Any, ...] | None = None
    degrees: tuple[Any, ...] | None = None
    positions: tuple[Any, ...] | None = None
    max_per_condition: int | None = None  # limit repeats per (mass, degree, position) condition
    min_trials: int = 6  # must have at least 6 trials to fit a 6x6 matrix

    def describe(self) -> str:
        return (
            f"{self.name} | masses={self.masses} degrees={self.degrees} "
            f"positions={self.positions} max_per_condition={self.max_per_condition}"
        )

dataclass(frozen=True)
class ProtocolSpecZone:
    name: str
    masses: tuple[Any, ...] | None = None
    degrees: tuple[Any, ...] | None = None
    positions: tuple[Any, ...] | None = None
    max_per_condition: int | None = None  # limit repeats per (mass, degree, position) condition
    min_trials: int = 6  # must have at least 6 trials to fit a 6x6 matrix
    zone_bins: int = 6

    def describe(self) -> str:
        return (
            f"{self.name} | masses={self.masses} degrees={self.degrees} "
            f"positions={self.positions} max_per_condition={self.max_per_condition}"
        )
