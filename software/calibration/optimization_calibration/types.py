from dataclasses import dataclass
import numpy as np

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
    deltaA_frob: np.ndarray
    cook_per_axis: np.ndarray
    cook_combined: np.ndarray

# @dataclass(frozen=True)
# class MonteCarloConfig:



    
