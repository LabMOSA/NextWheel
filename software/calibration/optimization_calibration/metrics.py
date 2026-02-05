import numpy as np

# Root Mean Square Error (RMSE) metrics
def RMSE_per_axis(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    """
    Compute the Root Mean Square Error (RMSE) per axis between true and predicted values.

    Parameters
    ----------
    Y : np.ndarray
        True values, shape (n_samples, n_axes).
    Y_pred : np.ndarray
        Predicted values, shape (n_samples, n_axes).

    Returns
    -------
    float
        RMSE per axis.
    """
    err = Y_pred - Y
    return np.sqrt(np.mean(err**2, axis=0))
def RMSE_total(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    """
    Compute the total Root Mean Square Error (RMSE) between true and predicted values.

    Parameters
    ----------
    Y : np.ndarray
        True values, shape (n_samples, n_axes).
    Y_pred : np.ndarray
        Predicted values, shape (n_samples, n_axes).

    Returns
    -------
    float
        Total RMSE.
    """
    err = Y_pred - Y
    print(err)
    print(np.mean(err**2))
    return float(np.sqrt(np.mean(err**2)))

# R-squared (R²) metrics
def R2_per_axis(Y: np.ndarray, Y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute the R-squared (R²) per axis between true and predicted values.

    Parameters
    ----------
    Y : np.ndarray
        True values, shape (n_samples, n_axes).
    Y_pred : np.ndarray
        Predicted values, shape (n_samples, n_axes).
    eps : float
        Small value to avoid division by zero.
    Returns
    -------
    float
        R² per axis.
    """
    err = Y_pred - Y
    ss_res = np.sum(err**2, axis=0)
    ss_tot = np.sum((Y - np.mean(Y, axis=0))**2, axis=0)
    # Avoid division by zero if an axis is (nearly) constant
    return np.where(ss_tot > eps, 1.0 - ss_res / ss_tot, np.nan)

def R2_total(Y: np.ndarray, Y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute the total R-squared (R²) between true and predicted values.

    Parameters
    ----------
    Y : np.ndarray
        True values, shape (n_samples, n_axes).
    Y_pred: np.ndarray
        Predicted values, shape (n_samples, n_axes).
    eps : float
        Small value to avoid division by zero.
    Returns
    -------
    float
        Total R².
    """
    err = Y_pred - Y
    ss_res_total = float(np.sum(err**2))
    ss_tot_total = float(np.sum((Y - np.mean(Y, axis=0))**2))

    return float(1.0 - ss_res_total / ss_tot_total) if ss_tot_total > eps else float("na")

def nrmse_per_axis(Y: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the Normalized Root Mean Square Error (NRMSE) per axis between true and predicted values.

    Parameters
    ----------
    Y : np.ndarray
        True values, shape (n_samples, n_axes).
    Y_pred : np.ndarray
        Predicted values, shape (n_samples, n_axes).

    Returns
    -------
    float
        NRMSE per axis.
    """
    rmse = RMSE_per_axis(Y, Y_pred)
    range_Y = np.ptp(Y, axis=0)  # Peak to peak (max - min) per axis
    return rmse / range_Y

