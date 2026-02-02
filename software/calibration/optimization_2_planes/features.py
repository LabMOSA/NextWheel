from typing import Sequence

import numpy as np
import re
from pathlib import Path

from software.calibration.optimization_calibration.types import (
    NormalizeConfig,
    BuildXYFn,
)
from software.calibration.fit_A_train import build_FMs_Channels
import software.calibration.wheelcalibration as wc


def compute_scale(Y: np.ndarray, configuration: NormalizeConfig) -> np.ndarray:
    """
    Compute scaling factors for each axis of Y based on the specified normalization configuration.
    Parameters
    ----------
    Y : np.ndarray
        Target matrix.
    configuration : NormalizeConfig
        Configuration for normalization.
    Returns
    -------
    scale : np.ndarray
        Scaling factors for each axis.
    """
    if not configuration.enabled:
        return np.ones(Y.shape[1], dtype=float)

    if configuration.method == "std":
        scale = np.std(Y, axis=0)
    elif configuration.method == "range":
        scale = np.max(Y, axis=0) - np.min(Y, axis=0)
    else:
        raise ValueError(f"Unknown normalization method: {configuration.method}")

    scale = np.where(scale < configuration.epsilon, 1.0, scale)
    return scale

def build_XY_core(
    force_trials: Sequence[dict],
    acc_bias: np.ndarray,
    base: np.ndarray,
    builder: BuildXYFn,
    normalize_y: NormalizeConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and target matrix Y from force trials.
    1. For each trial, compute the estimated forces/moments (Y) and
       the median of the analog force channels (X).
    2. Stack them into matrices.
    3. Optionally normalize Y.

    Parameters
    ----------
    force_trials : list of dict
        List of force trial data.
    acc_bias : np.ndarray
        Accelerometer bias.
    base : np.ndarray
        Base transformation matrix.
    builder : BuildXYFn
        Function to build X and Y matrices.
    normalize_y : NormalizeConfig | None, optional
        Configuration for normalizing Y, by default None.
    Returns
    -------
    X : np.ndarray
        Feature matrix.
    Y : np.ndarray
        Target matrix.
    """
    Y, X = builder(force_trials, acc_bias, base)
    Y = np.asarray(Y)
    X = np.asarray(X)
    y_scale = (
        compute_scale(Y, normalize_y)
        if normalize_y is not None
        else np.ones(Y.shape[1])
    )
    Y = Y / y_scale[None, :]
    return X, Y, y_scale

def build_XY(trials: Sequence[dict], acc_bias, base, normalize_y=None):
    return build_XY_core(trials, acc_bias, base, builder=build_FMs_Channels, normalize_y=normalize_y)




def position_from_imu_gravity(trial: dict) -> str | None:
    acc = None
    acc = trial["IMU"]["Acc"]
    if acc is None:
        return None
    acc = np.asarray(acc, float)  # shape (T,3)
    if acc.ndim != 2 or acc.shape[1] != 3:
        return None

    g = np.median(acc, axis=0)
    n = np.linalg.norm(g)
    if n < 1e-9:
        return None
    g = g / n

    ax = int(np.argmax(np.abs(g)))       # 0->X,1->Y,2->Z
    sign = "+" if g[ax] >= 0 else "-"
    axis = "XYZ"[ax]
    return f"{axis}{sign}"
