from pathlib import Path
from time import sleep

import numpy as np
from matplotlib import pyplot as plt

from software.calibration.optimization_2_planes.protocole import evaluate_protocol, protocol_score, search_best_protocols_from_indices, summarize_protocols
from software.calibration.optimization_calibration.PCA import (
    cap_trials_per_zone_mass_and_Y_ranges_no_pca,
    counts_per_axis_bin, default_score_fn,
)
from software.calibration.optimization_calibration.choice_groups import build_choice_groups_by_zone, groups_to_list, count_all_protocols, generate_random_protocols
from software.calibration.optimization_calibration.features import (
    position_from_imu_gravity,
    build_XY,
)
from software.calibration.optimization_calibration.fit import fit
from software.calibration.optimization_calibration.plot import (
    plot_pred_vs_true,
    plot_residuals,
)
from software.calibration.optimization_calibration.protocole import (
    recommend_protocol,
    select_trials,
    cap_trials_per_zone_and_mass,
)
from software.calibration.optimization_calibration import FitConfig, MonteCarloConfig
from software.calibration.cross_validation import (
    train_test_split_trials,
    load_force_trials,
)
from software.calibration.calculate_base_AccBias import load_fixed_imu_params
from software.calibration.optimization_calibration.types import ProtocolSpec, ProtocolEval

path = Path(__file__).resolve().parent.parent
imu_dir = path / "E1_E2"
packages_root = path / "package_trials_good"
trials = load_force_trials(packages_root)
base, acc_bias, imu_info = load_fixed_imu_params(imu_dir)


X_all, Y_all, _ = build_XY(trials, acc_bias, base)
X_all = np.asarray(X_all, float)


edges_fx = np.linspace(-100, 100, 9)
edges_fy = np.linspace(-100, 100, 9)
edges_fz = np.linspace(-100, 100, 9)

# Moments: souvent mieux avec peu de bins (ex 3 bins)
edges_mx = np.linspace(-20, 30, 4)
edges_my = np.linspace(-20, 30, 4)
edges_mz = np.linspace(-20, 30, 4)

edges_per_axis = [edges_fx, edges_fy, edges_fz, edges_mx, edges_my, edges_mz]
axes_for_ranges = [0, 1, 2, 3, 4, 5]

# Exemple d'utilisation
counts = counts_per_axis_bin(Y_all, edges_per_axis, axes_for_ranges)
for ax, c in counts.items():
    print(f"axe {ax} -> {c} (total={c.sum()})")

# y_tol = np.array([2.0, 2.0, 2.0, 20, 20, 20])  # à ajuster (N, N.m)
pct = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])      # 5% partout
min_tol = np.array([1, 1, 1, 0.5, 0.5, 0.5])     # plancher


selected = cap_trials_per_zone_mass_and_Y_ranges_no_pca(
    trials=trials,
    X=X_all,
    Y=Y_all,
    k_per_key=1,  # max 1 par (mass, zoneY) après dédoublonnage
    mass_fn=lambda t: t["Mass"],
    seed=44,
    edges_per_axis=[edges_fx, edges_fy, edges_fz, edges_mx, edges_my, edges_mz],
    axes_for_ranges=[0, 1, 2, 3, 4, 5],
    pct = pct,
    min_tol = min_tol,
    max_total=None,  # ou un entier si tu veux plafonner
)

for file in selected:
    print("Selected file:", file['__file__'])



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
print(len(X_sel))
Y_pred = X_sel @ fr.A.T
if getattr(fr, "b0", None) is not None:
    Y_pred = Y_pred + fr.b0[None, :]

print()
plot_pred_vs_true(Y_sel, Y_pred, title_prefix="Cap", split_forces_moments=True)
plot_residuals(Y_sel, Y_pred, title_prefix="Cap", split_forces_moments=True)

plt.show()

# kept_idx = margin_trials(
#     trials=trials,
#     Y=Y_all,
#     pct=pct,
#     min_tol=min_tol,
#     X=X_all,                 # optionnel
#     score_fn=default_score_fn, # optionnel
#     seed=44,
# )
# print(kept_idx)
# X_db, Y_db, _ = build_XY(kept_idx, acc_bias, base)
groups  = build_choice_groups_by_zone(
    trials = trials,
    X= X_all,
    Y=Y_all,
    seed=44,
    edges_per_axis=[edges_fx, edges_fy, edges_fz, edges_mx, edges_my, edges_mz],
    axes_for_ranges=[0, 1, 2, 3, 4, 5],
    pct = pct,
    min_tol = min_tol,
    max_candidates_per_group=1
)

protocols = generate_random_protocols(groups, n_protocols=10, seed=0)
chosen0 = [trials[i] for i in protocols[0]]
fit_cfg = FitConfig(method="ols", intercept=False)
mc_cfg = MonteCarloConfig(n_draws=300, frac=0.8, seed=42)
spec = ProtocolSpec(name= "Test Protocol")
# result_protocol: ProtocolEval = evaluate_protocol(chosen0, spec=spec, acc_bias=acc_bias, base=base, fit_cfg=fit_cfg, mc_cfg=mc_cfg)

best = search_best_protocols_from_indices(
    trial_pool=trials,
    protocol_indices=protocols,
    spec=spec,
    acc_bias=acc_bias,
    base=base,
    fit_cfg=fit_cfg,
    mc_cfg=mc_cfg,
    top_n=20,
    seed=0,
)
print(best)

rows = summarize_protocols(best)
print(rows)

# print(result_protocol.rmse_per_axis)
# print(result_protocol)
# print(f"Resultat du protocole: {protocol_score(result_protocol)}")

print(count_all_protocols(groups))
print(generate_random_protocols(groups, n_protocols=5))
print(f"Total groups: {len(groups)}")
print(f"Total trials: {len(trials)}")
for i, idxs in enumerate(groups.values()):
    print(f"Group {i}: {len(idxs)} trials")


# for i , trial in enumerate(trials):
#     pos = position_from_imu_gravity(trials[i])
#     print(f"Trial: {i}, File: {trial['__file__']}")
#     print(f"Position from IMU gravity: {pos}")
