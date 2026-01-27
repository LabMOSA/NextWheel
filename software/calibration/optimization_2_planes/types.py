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


@dataclass(frozen=True)
class PlaneSpec:
    y_idx: tuple[int, ...]
    name: str = ""


@dataclass(frozen=True)
class TwoPlaneProtocolSpec:
    name: str

    masses: tuple[Any, ...] | None = None
    degrees: tuple[Any, ...] | None = None
    positions: tuple[Any, ...] | None = None
    max_per_condition: int | None = None
    min_trials_per_plan: int = (
        3  # Minimum number of trials required per unique plane configuration
    )

    # Definition of the two planes
    plane_F_x_F_y_M_z: PlaneSpec = PlaneSpec(y_idx=(0, 1, 5), name="F_x, F_y, M_z")
    plane_M_x_M_y_F_z: PlaneSpec = PlaneSpec(y_idx=(3, 4, 2), name="F_y, M_x, M_z")

    def describe(self) -> str:
        return (
            f"{self.name} | masses={self.masses} degrees={self.degrees} "
            f"positions={self.positions} max_per_condition={self.max_per_condition}"
        )

@dataclass(frozen=True)
class MonteCarloConfig:
    n_draws: int = 300
    frac: float = 0.8
    seed: int = 0

@dataclass(frozen=True)
class CapConfig:
    k_per_key: int = 1
    pca_n_ang: int = 8
    pca_n_rad: int = 3

    # ranges (sur Y global) :
    axes_for_ranges: tuple[int, ...] | None = None
    edges_per_axis: tuple[tuple[float, ...], ...] | None = None

    max_per_pca_zone: int | None = None
    max_total: int | None = None


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
class PlaneEval:
    n_total: int
    n_train: int
    n_test: int
    rmse_total: float
    r2_total: float
    rmse_per_axis: list[float]
    r2_per_axis: list[float]
    notes: str = ""

@dataclass
class TwoPlaneProtocolEval:
    name: str
    plane_A: PlaneEval
    plane_B: PlaneEval

    # optionnel: score “global” si tu recombines A_final et testes sur un dataset mixte
    rmse_total_global: float | None = None
    r2_total_global: float | None = None

    notes: str = ""

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
