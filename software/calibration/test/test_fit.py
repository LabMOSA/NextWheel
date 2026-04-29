from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from software.calibration.cross_validation import (
    train_test_split_trials,
    load_force_trials,
)
from software.calibration.calculate_base_AccBias import load_fixed_imu_params
# from software.calibration.optimization_2_planes.outliers import remove_outliers_by_residual
from software.calibration.optimization_calibration.plot import plot_pred_vs_true, plot_residuals

from software.calibration.optimization_calibration import FitConfig
from software.calibration.optimization_calibration.features import build_XY, compute_scale
from software.calibration.optimization_calibration.fit import fit

from software.calibration.optimization_calibration.metrics import (
    rmse_total, r2_total, rmse_per_axis, r2_per_axis
)

# ----------------------------
# Paths / data loading
# ----------------------------
path = Path(__file__).resolve().parent.parent
imu_dir = path / "E1_E2"
packages_root = path / "package_trials_good"
trials = load_force_trials(packages_root)

base, acc_bias, imu_info = load_fixed_imu_params(imu_dir)

# ----------------------------
# Train/test split
# ----------------------------
train, test = train_test_split_trials(trials, test_ratio=0.2, seed=42, stratify_by_mass=True)

# ----------------------------
# Build raw X/Y (no scaling inside)
# IMPORTANT: we want to control scaling ourselves to avoid leakage
# ----------------------------
X_all, Y_all, _ = build_XY(trials, acc_bias, base, normalize_y=None)
X_train, Y_train, _ = build_XY(train, acc_bias, base, normalize_y=None)
X_test, Y_test, _ = build_XY(test, acc_bias, base, normalize_y=None)

# ----------------------------
# Compute y_scale on TRAIN ONLY
# ----------------------------
# Example: std scaling (recommended)
# If you already have NormalizeConfig, use it here. Otherwise:
y_scale = np.std(Y_train, axis=0)
y_scale = np.where(y_scale < 1e-12, 1.0, y_scale)

# Normalize targets using the SAME scale
Y_train_n = Y_train / y_scale[None, :]
Y_test_n  = Y_test  / y_scale[None, :]
Y_all_n   = Y_all   / y_scale[None, :]

# ----------------------------
# Fit in normalized space
# ----------------------------
fit_configuration = FitConfig(method="ols", intercept=False)
results = fit(X_train, Y_train_n, fit_configuration)

A = results.A
b0 = getattr(results, "b0", None)
out_dir = Path(__file__).resolve().parent.parent / "matrix_A_offset"
out_dir.mkdir(parents=True, exist_ok=True)

out_directory = out_dir / "calibration_matrix.npz"
np.savez(
    out_directory,
    A = A,
    b0 = b0,
    y_scale = y_scale,
)
print(f"y_scale: {y_scale}")
# ----------------------------
# Predict (normalized) then DE-normalize back to physical units
# Convention: Y_pred = X @ A.T (+ b0)
# ----------------------------
def predict_Y(X: np.ndarray) -> np.ndarray:
    Yp = X @ A.T
    if b0 is not None:
        Yp = Yp + b0[None, :]
    return Yp

Y_pred_test_n = predict_Y(X_test)
Y_pred_all_n  = predict_Y(X_all)

# DE-normalize
Y_pred_test = Y_pred_test_n * y_scale[None, :]
Y_pred_all  = Y_pred_all_n  * y_scale[None, :]

# ----------------------------
# Helpers: invalid predictions listing
# ----------------------------
from typing import Sequence, Any
AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

def list_invalid_predictions(
    Y: np.ndarray,
    Y_pred: np.ndarray,
    trials: Sequence[dict[str, Any]],
):
    invalid = ~np.isfinite(Y_pred)
    rows, cols = np.where(invalid)

    results = []
    for i, j in zip(rows, cols):
        trial = trials[i]
        results.append(
            {
                "idx": int(i),
                "axis": int(j),
                "axis_name": AXES[j] if j < len(AXES) else f"axis_{j}",
                "y_true": float(Y[i, j]),
                "y_pred": float(Y_pred[i, j]),
                "path": trial.get("__file__", None),
            }
        )
    return results

invalids = list_invalid_predictions(Y_all, Y_pred_all, trials)
print("Nb total de prédictions invalides:", len(invalids))
for r in invalids[:50]:
    print(
        f"[{r['idx']:3d}] {r['axis_name']} | true={r['y_true']:7.2f} "
        f"| pred={r['y_pred']} | path={r['path']}"
    )

# ----------------------------
# Metrics (always in PHYSICAL units)
# ----------------------------
rmse_total = rmse_total(Y_all, Y_pred_all)
rmse_per_axis = rmse_per_axis(Y_all, Y_pred_all)

r2_total = r2_total(Y_all, Y_pred_all)
r2_per_axis = r2_per_axis(Y_all, Y_pred_all)

print("RMSE total:", rmse_total)
print("RMSE axes :", rmse_per_axis)
print("R2 total  :", r2_total)
print("R2 axes   :", r2_per_axis)

# ----------------------------
# Plots (physical units)
# ----------------------------
plot_pred_vs_true(Y_all, Y_pred_all, title_prefix="Calibration — Test (denorm)", split_forces_moments=True)
plot_residuals(Y_all, Y_pred_all, title_prefix="Calibration — Test (denorm)", split_forces_moments=True)
plt.show()

# ----------------------------
# Worst residual trials (physical units) + ignore NaNs
# ----------------------------
def get_worst_residual_trials_safe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trials: Sequence,
    axis: int,
    top_k: int = 10,
):
    residuals = y_pred[:, axis] - y_true[:, axis]
    abs_res = np.abs(residuals)

    finite = np.isfinite(abs_res)
    finite_idx = np.where(finite)[0]
    if finite_idx.size == 0:
        return []

    worst_idx = finite_idx[np.argsort(abs_res[finite])[-top_k:]][::-1]

    results = []
    for i in worst_idx:
        results.append((int(i), float(residuals[i]), float(y_true[i, axis]), float(y_pred[i, axis]), trials[i]))
    return results

worst_mx = get_worst_residual_trials_safe(
    y_true=Y_all,
    y_pred=Y_pred_all,
    trials=trials,
    axis= 3,
    top_k=10,
)

worst_my = get_worst_residual_trials_safe(
    y_true=Y_all,
    y_pred=Y_pred_all,
    trials=trials,
    axis= 4,
    top_k=10,
)
for i, r, yt, yp, trial in worst_mx:
    print(f"Trial {trial['__file__']}")
    print(f"Trial {i:3d} | Mx true = {yt:7.2f} | Mx pred = {yp:7.2f} | residual = {r:7.2f}")

for i, r, yt, yp, trial in worst_my:
    print(f"Trial {trial['__file__']}")
    print(f"Trial {i:3d} | My true = {yt:7.2f} | My pred = {yp:7.2f} | residual = {r:7.2f}")

# y_true_f, y_pred_f, trials_f, removed = remove_outliers_by_residual(
#     y_true=Y_all,
#     y_pred=Y_pred_all,
#     trials=trials,
#     axis=None,              # <-- tous les axes (norme L2)
#     method="quantile",
#     quantile=0.98,          # top 2%
# )

AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

def print_removed_outliers(removed, y_true, y_pred, trials, top=50):
    resid = y_pred - y_true
    for k, r in enumerate(removed[:top]):
        i = r["idx"]
        v = resid[i]
        # axes qui contribuent le plus (tri par |résidu|)
        order = np.argsort(np.abs(v))[::-1]
        top_axes = ", ".join(
            f"{AXES[j]}={v[j]:+.2f}" for j in order[:3]
        )
        print(
            f"[{k+1:02d}] idx={i:3d} score(L2)={r['score']:.3f} "
            f"path={trials[i].get('__file__','<unknown>')} | top: {top_axes}"
        )
