from pathlib import Path
import numpy as np

from software.calibration.optimization_calibration.export import analyze_selected_protocols
from software.calibration.optimization_calibration.fit import fit
from software.calibration.optimization_calibration.plot import plot_pred_vs_true, plot_residuals
from software.calibration.optimization_calibration.protocole import search_best_protocols_from_indices, summarize_protocols
from software.calibration.optimization_calibration.choice_groups import build_choice_groups_by_zone, generate_random_protocols
from software.calibration.optimization_calibration.features import build_XY
from software.calibration.optimization_calibration import FitConfig, MonteCarloConfig
from software.calibration.cross_validation import (
    load_force_trials,
)
from software.calibration.calculate_base_AccBias import load_fixed_imu_params
from software.calibration.optimization_calibration.recommand_protocol import print_best_protocol_trials
from software.calibration.optimization_calibration.report import select_plateau
from software.calibration.optimization_calibration.calibration_types import ProtocolSpec

path = Path(__file__).resolve().parent.parent
imu_dir = path / "E1_E2"
packages_root = path / "trials_organized"
trials = load_force_trials(packages_root)
base, acc_bias, imu_info = load_fixed_imu_params(imu_dir)


X_all, Y_all, _ = build_XY(trials, acc_bias, base)
X_all = np.asarray(X_all, float)


edges_fx = np.linspace(-100, 120, 9)
edges_fy = np.linspace(-100, 120, 9)
edges_fz = np.linspace(-100, 100, 9)

edges_mx = np.linspace(-20, 30, 4)
edges_my = np.linspace(-20, 30, 4)
edges_mz = np.linspace(-20, 30, 4)

edges_per_axis = [edges_fx, edges_fy, edges_fz, edges_mx, edges_my, edges_mz]
axes_for_ranges = [0, 1, 2, 3, 4, 5]

pct = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
min_tol = np.array([1, 1, 1, 0.5, 0.5, 0.5])




# --- 6 graphs: one per channel (hist overlay) ---
axis_names = ["Ch0", "Ch1", "Ch2", "Ch3", "Ch4", "Ch5"]


groups  = build_choice_groups_by_zone(
    trials = trials,
    X= X_all,
    Y=Y_all,
    seed=44,
    edges_per_axis=[edges_fx, edges_fy, edges_fz, edges_mx, edges_my, edges_mz],
    axes_for_ranges=[0, 1, 2, 3, 4, 5],
    pct = pct,
    min_tol = min_tol
)

protocols = generate_random_protocols(groups, n_protocols= 5, seed=0)
chosen0 = [trials[i] for i in protocols[0]]
fit_cfg = FitConfig(method="ols", intercept=False)
mc_cfg = MonteCarloConfig(n_draws=300, frac=0.8, seed=42)
spec = ProtocolSpec(name= "Test Protocol")


best = search_best_protocols_from_indices(
    trial_pool=trials,
    protocol_indices=protocols,
    spec=spec,
    acc_bias=acc_bias,
    base=base,
    fit_configuration= fit_cfg,
    montecarlo_configuration= mc_cfg,
    top_n=20
)
print(best)

rows = summarize_protocols(best)
print(rows)

# Plot best protocol predictions vs true values, and residuals
best_protocol_indices = best[0][2]
print(best_protocol_indices)
chosen_trials = [trials[i] for i in best_protocol_indices]
X_test, Y_test, _ = build_XY(chosen_trials, acc_bias, base)
X_test = np.asarray(X_test, float)
# Fit on the chosen protocol trials
fit_results = fit(X_test, Y_test, fit_cfg)
A = fit_results.A
print(A)
Y_pred_test = X_test @ A.T

plot_pred_vs_true(Y_test, Y_pred_test, title_prefix="Calibration — Test (denorm)", split_forces_moments=True)
plot_residuals(Y_test, Y_pred_test, title_prefix="Calibration — Test (denorm)", split_forces_moments=True)
print_best_protocol_trials(best, trials, max_protocols=100, show_file=False)

best_plateau = select_plateau(best, portion_kept=0.2, max_kept=10)

df = analyze_selected_protocols(
    best=best_plateau,
    trial_pool=trials,
    protocol_limit=None,    # already limited by plateau
    merge="union",
    angle_bin_deg=10.0,
)





