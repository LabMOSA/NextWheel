from software.calibration.optimization_calibration.types import FitConfig, FitResult
import numpy as np

def _argument_intercept(X: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack([ones, X])

def fit(X: np.ndarray, Y: np.ndarray, configuration: FitConfig):
    """ """
    X_copy = X
    if configuration.intercept:
        _argument_intercept(X)

    if configuration.method == "ols":
        B, *_ = np.linalg.lstsq(X_copy, Y, rcond=None)
    elif configuration.method == "ridge":
        alpha = configuration.alpha_ridge
        XtX = X_copy.T @ X_copy
        p = XtX.shape[0]
        ridge_term = alpha * np.eye(p)
        B = np.linalg.solve(XtX + ridge_term, X_copy.T @ Y)
    else:
        raise ValueError(f"Unknown fitting method: {configuration.method}")

    if configuration.intercept:
        b0 = B[0, :]
        A = B[1:, :].T
    else:
        b0 = None
        A = B.T

    return FitResult(A=A, b0=b0, B=B)


