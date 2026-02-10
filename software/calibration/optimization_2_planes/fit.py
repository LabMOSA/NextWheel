from software.calibration.optimization_2_planes.features import build_XY
from software.calibration.optimization_2_planes.types import (
    FitConfig,
    FitResult,
    FitResultTwoPlanes,
)
import numpy as np
from software.calibration.optimization_2_planes.organize_trials import assign_planes

IDX = {"Fx": 0, "Fy": 1, "Fz": 2, "Mx": 3, "My": 4, "Mz": 5}
CHANNELS = {"Ch0": 0, "Ch1": 1, "Ch2": 2, "Ch3": 3, "Ch4": 4, "Ch5": 5}
PLANE_A_IDX = [IDX["Fx"], IDX["Fy"], IDX["Mz"]]  # Fx,Fy,Mz
PLANE_A_CHANNELS = [CHANNELS["Ch0"], CHANNELS["Ch2"], CHANNELS["Ch4"]]
PLANE_B_IDX = [IDX["Mx"], IDX["My"], IDX["Fz"]]  # Mx,My,Fz
PLANE_B_CHANNELS = [CHANNELS["Ch1"], CHANNELS["Ch3"], CHANNELS["Ch5"]]

def _argument_intercept(X: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack([ones, X])



def fit_plane_3x3(
    X: np.ndarray, Y: np.ndarray, in_idx: list[int], out_idx: list[int], cfg
):
    """
    Fit only Y[:, out_idx] from X[:, in_idx].
    Returns A_plane (3x3) and b0_plane (3,) if intercept.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)

    if len(in_idx) != 3 or len(out_idx) != 3:
        raise ValueError("This helper is for 3x3 only (need 3 in_idx and 3 out_idx).")

    Xs = X[:, in_idx]  # (n,3)
    Ys = Y[:, out_idx]  # (n,3)

    # intercept (optional)
    if cfg.intercept:
        Xs_aug = np.hstack([np.ones((Xs.shape[0], 1)), Xs])  # (n,4)
    else:
        Xs_aug = Xs  # (n,3)

    if cfg.method == "ols":
        B, *_ = np.linalg.lstsq(Xs_aug, Ys, rcond=None)  # (3 or 4, 3)
    elif cfg.method == "ridge":
        alpha = cfg.alpha_ridge
        XtX = Xs_aug.T @ Xs_aug
        p = XtX.shape[0]
        B = np.linalg.solve(XtX + alpha * np.eye(p), Xs_aug.T @ Ys)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    if cfg.intercept:
        b0 = B[0, :]  # (3,)
        A_plane = B[1:, :].T  # (3,3)
    else:
        b0 = None
        A_plane = B.T  # (3,3)
    return A_plane, b0, B


def fit_two_planes(
    trials, acc_bias, base, configuration: FitConfig
) -> FitResultTwoPlanes:
    # Split trials
    out = assign_planes(trials)
    trials_plane_A = out["plane_FxFyMz"]
    trials_plane_B = out["plane_MxMyFz"]

    print(f"Path trials Plane A: {[t.get('__file__','<unknown>') for t in trials_plane_A]}")
    print(f"Number of trials Plane A: {len(trials_plane_A)}")
    print (f"Path trials Plane B: {[t.get('__file__','<unknown>') for t in trials_plane_B]}")
    print(f"Number of trials Plane B: {len(trials_plane_B)}")
    if len(trials_plane_A) < 3:
        raise ValueError("Not enough trials in Plane A to perform fitting.")
    if len(trials_plane_B) < 3:
        raise ValueError("Not enough trials in Plane B to perform fitting.")

    # Build X and Y for Plane A
    X_plane_A, Y_plane_A, _ = build_XY(trials_plane_A, acc_bias, base)
    X_plane_B, Y_plane_B, _ = build_XY(trials_plane_B, acc_bias, base)

    # Fit for Plane A (Fx, Fy, Mz)
    A_A, b0_A, B_A = fit_plane_3x3(
        X_plane_A,
        Y_plane_A,
        in_idx=PLANE_A_CHANNELS,
        out_idx= PLANE_A_IDX,
        cfg=configuration,
    )
    fit_result_A: FitResult = FitResult(A=A_A, b0=b0_A, B=B_A)

    # Fit for Plane B (Mx, My, Fz)
    A_B, b0_B, B_B = fit_plane_3x3(
        X_plane_B,
        Y_plane_B,
        in_idx=PLANE_B_CHANNELS,
        out_idx=PLANE_B_IDX,
        cfg=configuration,
    )
    fit_result_B: FitResult = FitResult(A=A_B, b0=b0_B, B=B_B)

    # Reconstruct full A matrix
    A_full = np.zeros((6, 6), dtype=float)
    A_full[np.ix_(PLANE_A_IDX, PLANE_A_CHANNELS)] = fit_result_A.A
    A_full[np.ix_(PLANE_B_IDX, PLANE_B_CHANNELS)] = fit_result_B.A

    # Reconstruct full b0 vector if intercept is used
    b0_full = np.zeros((6,), dtype=float) if configuration.intercept else None
    if configuration.intercept:
        b0_full[PLANE_A_IDX] = fit_result_A.b0
        b0_full[PLANE_B_IDX] = fit_result_B.b0
    return FitResultTwoPlanes(
        A_full=A_full,
        b0_full=b0_full,
        res_plane_A=fit_result_A,
        res_plane_B=fit_result_B,
    )
