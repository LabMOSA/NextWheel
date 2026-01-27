from pathlib import Path
from time import sleep

import numpy as np
from matplotlib import pyplot as plt

from software.calibration.optimization_calibration.PCA import cap_trials_per_zone_mass_pca_and_ranges
from software.calibration.optimization_calibration.features import position_from_imu_gravity, build_XY
from software.calibration.optimization_calibration.fit import fit
from software.calibration.optimization_calibration.plot import plot_pred_vs_true, plot_residuals
from software.calibration.optimization_calibration.protocole import recommend_protocol, select_trials, cap_trials_per_zone_and_mass
from software.calibration.optimization_calibration import FitConfig, MonteCarloConfig
from software.calibration.cross_validation import (
    train_test_split_trials,
    load_force_trials,
)
from software.calibration.calculate_base_AccBias import load_fixed_imu_params

path = Path(__file__).resolve().parent.parent
imu_dir = path / "E1_E2"
packages_root = path / "package_trials_good"
trials = load_force_trials(packages_root)
base, acc_bias, imu_info = load_fixed_imu_params(imu_dir)

# X_all, _, _ = build_XY(trials, acc_bias, base)
# for i, trial in enumerate(trials):
#     seed = 42
#     selected_trials = cap_trials_per_zone_and_mass(trials, X = X_all, k_per_key= 1, mass_fn= lambda t: t['Mass'], seed=seed, n_bins=6, zone_id=None)
#     print(selected_trials)
# --- Build X for ALL trials (needed for zoning) ---
X_all, Y_all, _ = build_XY(trials, acc_bias, base)
X_all = np.asarray(X_all, float)

edges_fx = np.linspace(-40, 40, 9)
edges_fy = np.linspace(-40, 40, 9)
edges_fz = np.linspace(0, 200, 9)

# Moments: souvent mieux avec peu de bins (ex 3 bins)
edges_mx = np.array([-np.inf, -0.5, 0.5, np.inf])
edges_my = np.array([-np.inf, -0.5, 0.5, np.inf])
edges_mz = np.array([-np.inf, -0.5, 0.5, np.inf])

selected = cap_trials_per_zone_mass_pca_and_ranges(
    trials=trials,
    X=X_all,
    Y=Y_all,
    k_per_key=1,
    mass_fn=lambda t: t["Mass"],
    seed=42,
    pca_n_ang=7,
    pca_n_rad=4,
    edges_per_axis=[edges_fx, edges_fy, edges_fz, edges_mx, edges_my, edges_mz],
    axes_for_ranges=[0, 1, 2, 3, 4, 5],
    max_per_pca_zone= 3,
)

# --- Build X for SELECTED trials ---
X_sel, Y_sel, _ = build_XY(selected, acc_bias, base)
X_sel = np.asarray(X_sel, float)

print(f"Total trials: {len(trials)}")
print(f"Selected trials: {len(selected)}")

# --- 6 graphs: one per channel (hist overlay) ---
axis_names = ["Ch0", "Ch1", "Ch2", "Ch3", "Ch4", "Ch5"]  # renomme si tu veux

for j in range(6):
    plt.figure()
    plt.hist(X_all[:, j], bins=40, density=True, alpha=0.5, label="All trials")
    plt.hist(X_sel[:, j], bins=40, density=True, alpha=0.5, label="Selected trials")
    plt.title(f"Distribution of X channel {j} ({axis_names[j]})")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

fit_cfg = FitConfig(method="ols", intercept=False)
mc_cfg = MonteCarloConfig(n_draws=300, frac=0.8, seed=42)

fr = fit(X_sel, Y_sel, fit_cfg)

Y_pred = X_sel @ fr.A.T
if getattr(fr, "b0", None) is not None:
    Y_pred = Y_pred + fr.b0[None, :]

plot_pred_vs_true(Y_sel, Y_pred, title_prefix="Cap", split_forces_moments=True)
plot_residuals(Y_sel, Y_pred, title_prefix="Cap", split_forces_moments=True)

plt.show()


# for i , trial in enumerate(trials):
#     pos = position_from_imu_gravity(trials[i])
#     print(f"Trial: {i}, File: {trial['__file__']}")
#     print(f"Position from IMU gravity: {pos}")



