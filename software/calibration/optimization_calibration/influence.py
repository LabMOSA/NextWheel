import numpy as np
from .calibration_types import InfluenceConfig, InfluenceResult


def compute_predictions(
        X: np.ndarray, A: np.ndarray, b0: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute predicted values Y_pred from design matrix X and coefficients A.

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (n_samples, n_predictors).

    A : np.ndarray
        Coefficient matrix, shape (n_predictors, n_axes).

    b0 : np.ndarray | None
        Intercept vector, shape (n_axes,). If None, no intercept is added.

    Returns
    -------
    np.ndarray
        Predicted values Y_pred, shape (n_samples, n_axes).
    """
    Y_hat = X @ A
    if b0 is not None:
        Y_hat += b0[None, :]
    return Y_hat


def compute_residuals(Y: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """
    Compute residuals between true and predicted values.

    Parameters
    ----------
    Y : np.ndarray
        True values, shape (n_samples, n_axes).
    Y_pred : np.ndarray
        Predicted values, shape (n_samples, n_axes).

    Returns
    -------
    np.ndarray
        Residuals, shape (n_samples, n_axes).
    """
    return Y_pred - Y


def compute_H(X: np.ndarray) -> np.ndarray:
    """
    Compute hat matrix H from design matrix X and (X^T X)^-1.

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (n_samples, n_predictors).

    Returns
    -------
    np.ndarray
        Hat matrix H, shape (n_samples, n_samples).
    """
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    return X @ XtX_inv @ X.T


def compute_leverage(H: np.ndarray) -> np.ndarray:
    """
    Return leverage values h_ii from the hat matrix H.

    Parameters
    ----------
    H : np.ndarray
        Hat matrix, shape (n_samples, n_samples).

    Returns
    -------
    np.ndarray
        Leverage values (diagonal of H), shape (n_samples,).
    """
    H = np.asarray(H, float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square (n,n). Got shape {H.shape}.")
    return np.diag(H)


def compute_denominator_from_leverage(h: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """
    Compute denominator values (1 - h) with numerical stability.

    Parameters
    ----------
    h : np.ndarray
        Leverage values, shape (n_samples,).

    epsilon : float
        Small value to avoid division by zero.

    Returns
    -------
    np.ndarray
        Denominator values (1 - h), shape (n_samples,).
    """
    denominator = 1.0 - h
    denominator = np.maximum(denominator, epsilon)
    return denominator


def compute_loo_residuals(E: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Compute leave-one-out (LOO) residuals.

    Parameters
    ----------
    E : np.ndarray
        Residuals, shape (n_samples, n_axes).

    denominator : np.ndarray
        Denominator values (1 - h), shape (n_samples,).

    Returns
    -------
    np.ndarray
        LOO residuals, shape (n_samples, n_axes).
    """
    return E / denominator[:, None]


def compute_U(X: np.ndarray, use_pseudo_inverse: bool = True) -> np.ndarray:
    """
    Compute matrix U from design matrix X.

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (n_samples, n_predictors).

    use_pseudo_inverse : bool
        Whether to use pseudo-inverse for numerical stability.

    Returns
    -------
    np.ndarray
        Matrix U, shape (n_samples, n_predictors).
    """
    XtX = X.T @ X
    if use_pseudo_inverse:
        XtX_inv = np.linalg.pinv(XtX)
    else:
        XtX_inv = np.linalg.inv(XtX)
    return (XtX_inv @ X.T).T


def compute_deltaA_Frobenius(
        U: np.ndarray, E: np.ndarray, denominator: np.ndarray
) -> np.ndarray:
    """ """
    N = E.shape[0]
    deltaA_Frobenius = np.empty(N, dtype=float)
    for i in range(N):
        deltaB = -np.outer(U[i], E[i]) / denominator[i]
        deltaA = deltaB.T
        deltaA_Frobenius[i] = np.linalg.norm(deltaA, "fro")
    return deltaA_Frobenius


def compute_cook_like(
        E: np.ndarray,
        h: np.ndarray,
        denominator: np.ndarray,
        p: int,
        mse: np.ndarray,
        epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Cook's distance-like metrics per axis and combined.

    Parameters

    ----------
    E : np.ndarray
        Residuals, shape (n_samples, n_axes).

    h : np.ndarray
        Leverage values, shape (n_samples,).

    denominator : np.ndarray
        Denominator values (1 - h), shape (n_samples,).

    p : int
        Number of predictors.
    mse : np.ndarray
        Mean squared error per axis, shape (n_axes,).

    epsilon : float
        Small value to avoid division by zero.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Cook's distance per axis and combined.
    """
    cook = (
            (E ** 2) / (p * (mse[None, :] + epsilon)) * (h[:, None] / (denominator[:, None] ** 2))
    )
    cook_combined = np.mean(cook, axis=1)
    return cook, cook_combined


def compute_mse_per_axis(E: np.ndarray, dof: int, eps: float = 1e-12) -> np.ndarray:
    """
    Compute mean squared error (MSE) per axis with numerical stability.

    Parameters
    ----------
    E : np.ndarray
        Residuals, shape (n_samples, n_axes).
    dof : int
        Degrees of freedom.
    eps : float
        Small value to avoid division by zero.
    Returns
    -------
    np.ndarray
        MSE per axis, shape (n_axes,).
    """
    mse = np.sum(E ** 2, axis=0) / max(dof, 1)
    return np.where(mse > eps, mse, eps)


def influence_analytic(
        X: np.ndarray,
        Y: np.ndarray,
        A: np.ndarray,
        b0: np.ndarray | None = None,
        influence_configuration: InfluenceConfig = InfluenceConfig(),
) -> InfluenceResult:
    """
    Compute analytic influence metrics for linear regression.
    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (n_samples, n_predictors).
    Y : np.ndarray
        True values, shape (n_samples, n_axes).
    A : np.ndarray
        Coefficient matrix, shape (n_predictors, n_axes).
    b0 : np.ndarray | None
        Intercept vector, shape (n_axes,). If None, no intercept is used.
    influence_configuration : InfluenceConfig
        Configuration for influence computation.
    Returns
    -------
    InfluenceResult
        A dataclass containing leverage, residuals, LOO residuals,
        ΔA Frobenius norm, Cook's distance per axis, and combined Cook's distance
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    N, p = X.shape

    # 1) Yhat + residuals
    Yhat = compute_predictions(X, A, b0=b0)
    E = compute_residuals(Y, Yhat)

    # 2) Compute H
    H = compute_H(X)

    # 3) leverage + denom
    h = compute_leverage(H)
    denominator = compute_denominator_from_leverage(h, epsilon=influence_configuration.epsilon)

    # 4) LOO residuals
    E_loo = compute_loo_residuals(E, denominator)

    # 5) ΔA score
    U = compute_U(X)
    deltaA_Frobenius = compute_deltaA_Frobenius(E, U, denominator)

    # 6) Cook-like
    dof = max(N - p, 1)
    mse = compute_mse_per_axis(E, dof=dof, eps=influence_configuration.epsilon)
    cook_per_axis, cook_combined = compute_cook_like(
        E=E, h=h, denominator=denominator, p=p, mse=mse, epsilon=influence_configuration.epsilon
    )

    return InfluenceResult(
        leverage=h,
        residuals=E,
        loo_residuals=E_loo,
        delta_A_Frobenius=deltaA_Frobenius,
        cook_per_axis=cook_per_axis,
        cook_combined=cook_combined,
    )
