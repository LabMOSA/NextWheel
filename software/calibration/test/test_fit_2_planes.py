from pathlib import Path

import matplotlib.pyplot as plt

from software.calibration.cross_validation import (
    train_test_split_trials,
    load_force_trials,
)
from software.calibration.calculate_base_AccBias import load_fixed_imu_params
from software.calibration.optimization_2_planes.features import build_XY
from software.calibration.optimization_2_planes.fit import fit_two_planes
from software.calibration.optimization_2_planes.outliers import get_worst_residual_trials, remove_outliers_by_residual
from software.calibration.optimization_2_planes.types import FitResultTwoPlanes
from software.calibration.optimization_2_planes import FitConfig
from software.calibration.optimization_calibration.metrics import (
    rmse_total, r2_total, rmse_per_axis, r2_per_axis
)
from software.calibration.optimization_calibration.plot import (
    plot_pred_vs_true,
    plot_residuals,
)

# ----------------------------
# Paths / data loading
# ----------------------------
path = Path(__file__).resolve().parent.parent          # ../calibration
imu_dir = path / "E1_E2"
packages_root = path / "package_trials_good"          # attention au nom exact du dossier
trials = load_force_trials(packages_root)

base, acc_bias, imu_info = load_fixed_imu_params(imu_dir)

# ----------------------------
# Fit
# ----------------------------
train_trials, test_trials = train_test_split_trials(trials,test_ratio=0.2, seed=42, stratify_by_mass=True)

X_, Y_, _ = build_XY(trials, acc_bias, base)
X_train, Y_train, _ = build_XY(train_trials, acc_bias, base)
fit_configuration = FitConfig(method="ols", intercept=False)
fit_2_plans: FitResultTwoPlanes = fit_two_planes(train_trials, acc_bias, base, fit_configuration)

X_test, Y_test, _ = build_XY(test_trials, acc_bias, base)

# ----------------------------
# Predict on test
# Convention: Y_pred = X @ A.T (+ b0)
# ----------------------------
A_full = fit_2_plans.A_full
b0 = fit_2_plans.b0_full

Y_pred = (X_test @ A_full.T)
if b0 is not None:
    Y_pred = Y_pred + b0[None, :]
#
#
# # ----------------------------
# # Train Metrics (for information)
# # ----------------------------
#
Y_train_pred = (X_train @ A_full.T)
if b0 is not None:
    Y_train_pred = Y_train_pred + b0[None, :]

Y_pred_all_trials = (X_ @ A_full.T)
if b0 is not None:
    Y_pred_all_trials = Y_pred_all_trials + b0[None, :]

rmse_total_train = rmse_total(Y_train, Y_train_pred)
print("RMSE total train:", rmse_total_train)
rmse_per_axis = rmse_per_axis(Y_train, Y_train_pred)

print("RMSE axes train :", rmse_per_axis)

r2_total_train = r2_total(Y_train, Y_train_pred)
r2_per_axis_train = r2_per_axis(Y_train, Y_train_pred)

print("R2 total train  :", r2_total_train)
print("R2 axes train   :", r2_per_axis_train)

#
# ----------------------------
# Train Plots (for information)
# ----------------------------
plot_pred_vs_true(Y_train, Y_train_pred, title_prefix="Calibration — Train", split_forces_moments=True)
plot_residuals(Y_train, Y_train_pred, title_prefix="Calibration — Train", split_forces_moments=True)
# ----------------------------
# Plots
# ----------------------------
plot_pred_vs_true(Y_test, Y_pred, title_prefix="Calibration — Test", split_forces_moments=True)
plot_residuals(Y_test, Y_pred, title_prefix="Calibration — Test", split_forces_moments=True)

plt.show()
#

#
worst_fy = get_worst_residual_trials(
    y_true=Y_train,
    y_pred= Y_train_pred,
    trials=trials,
    axis= 1,   # Fy
    top_k=30,
)

for i, r, yt, yp, trial in worst_fy:
    print(f"Trial {trial['__file__']}")
    print(
        f"Trial {i:3d} | Fy true = {yt:7.2f} | Fy pred = {yp:7.2f} | residual = {r:7.2f}"
    )


# # ----------------------------
# # Metrics
# # ----------------------------
# rmse_total = RMSE_total(Y_test, Y_pred)
# rmse_per_axis = RMSE_per_axis(Y_test, Y_pred)
#
# r2_total = R2_total(Y_test, Y_pred)
# r2_per_axis = R2_per_axis(Y_test, Y_pred)
#
# print("RMSE total:", rmse_total)
# print("RMSE axes :", rmse_per_axis)
# print("R2 total  :", r2_total)
# print("R2 axes   :", r2_per_axis)
#





