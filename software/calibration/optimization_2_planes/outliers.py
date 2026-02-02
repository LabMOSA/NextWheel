import numpy as np
from typing import Sequence, Any


def get_worst_residual_trials(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trials: Sequence,
    axis: int,
    top_k: int = 10,
):
    """
    Return the trials with the largest absolute residuals on a given axis.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples, n_axes)
    y_pred : np.ndarray, shape (n_samples, n_axes)
    trials : Sequence
        Original trials (dicts or objects), same order as y_true.
    axis : int
        Axis index (e.g. 1 for Fy).
    top_k : int
        Number of worst trials to return.

    Returns
    -------
    list of tuples
        (index, residual, y_true, y_pred, trial)
    """
    residuals = y_pred[:, axis] - y_true[:, axis]
    abs_res = np.abs(residuals)

    worst_idx = np.argsort(abs_res)[-top_k:][::-1]

    results = []
    for i in worst_idx:
        results.append(
            (
                i,
                residuals[i],
                y_true[i, axis],
                y_pred[i, axis],
                trials[i],
            )
        )

    return results


def remove_outliers_by_residual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trials: Sequence[Any],
    axis: int | None = None,
    *,
    method: str = "quantile",   # "top_k" | "quantile" | "sigma"
    top_k: int = 10,
    quantile: float = 0.98,     # keep up to this quantile of |res|
    n_sigma: float = 3.0,
):
    """
    Remove outlier trials based on residual magnitude.

    Parameters
    ----------
    axis : int | None
        If int: use only that axis residuals (e.g., 1 for Fy).
        If None: use L2 norm of residual vector across all axes.
    method : str
        "top_k": removes the top_k largest residuals
        "quantile": removes samples with |res| > quantile(|res|)
        "sigma": removes samples with |res| > mean(|res|) + n_sigma*std(|res|)

    Returns
    -------
    y_true_f, y_pred_f, trials_f, removed
        removed is a list of dicts with keys:
        idx, score, residual (scalar or vector), path
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have same shape"
    assert len(trials) == y_true.shape[0], "trials must align with y_true rows"

    resid = y_pred - y_true

    if axis is None:
        score = np.linalg.norm(resid, axis=1)  # global outlier score
    else:
        score = np.abs(resid[:, axis])         # axis-specific score

    if method == "top_k":
        k = int(min(max(top_k, 0), len(score)))
        removed_idx = np.argsort(score)[-k:]
    elif method == "quantile":
        q = float(quantile)
        if not (0.0 < q < 1.0):
            raise ValueError("quantile must be in (0, 1)")
        thr = np.quantile(score, q)
        removed_idx = np.where(score > thr)[0]
    elif method == "sigma":
        thr = float(score.mean() + n_sigma * score.std(ddof=1))
        removed_idx = np.where(score > thr)[0]
    else:
        raise ValueError("method must be one of: 'top_k', 'quantile', 'sigma'")

    removed_idx = np.unique(removed_idx).astype(int)
    keep_mask = np.ones(len(score), dtype=bool)
    keep_mask[removed_idx] = False

    y_true_f = y_true[keep_mask]
    y_pred_f = y_pred[keep_mask]
    trials_f = [t for i, t in enumerate(trials) if keep_mask[i]]

    removed = []
    for i in removed_idx:
        trial = trials[i]
        removed.append(
            {
                "idx": int(i),
                "score": float(score[i]),
                "residual": resid[i].copy() if axis is None else float((y_pred[i, axis] - y_true[i, axis])),
                "path": trial.get("__file__", "<unknown>"),
            }
        )

    # sort removed from worst to less-worst
    removed.sort(key=lambda d: d["score"], reverse=True)

    return y_true_f, y_pred_f, trials_f, removed