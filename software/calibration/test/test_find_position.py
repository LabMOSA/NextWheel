from pathlib import Path
import numpy as np

from software.calibration.optimization_calibration.protocole import evaluate_protocol, protocol_score, search_best_protocols_from_indices, summarize_protocols
from software.calibration.optimization_calibration.choice_groups import build_choice_groups_by_zone, groups_to_list, count_all_protocols, generate_random_protocols
from software.calibration.optimization_calibration.features import (
    build_XY,
)
from software.calibration.optimization_calibration import FitConfig, MonteCarloConfig
from software.calibration.cross_validation import (
    load_force_trials,
)
from software.calibration.calculate_base_AccBias import load_fixed_imu_params
from software.calibration.optimization_calibration.recommand_protocol import print_best_protocol_trials
from software.calibration.optimization_calibration.types import ProtocolSpec

path = Path(__file__).resolve().parent.parent
imu_dir = path / "E1_E2"
packages_root = path / "trials_organized"
trials = load_force_trials(packages_root)
base, acc_bias, imu_info = load_fixed_imu_params(imu_dir)


X_all, Y_all, _ = build_XY(trials, acc_bias, base)
X_all = np.asarray(X_all, float)


edges_fx = np.linspace(-100, 100, 9)
edges_fy = np.linspace(-100, 100, 9)
edges_fz = np.linspace(-100, 100, 9)

edges_mx = np.linspace(-20, 30, 4)
edges_my = np.linspace(-20, 30, 4)
edges_mz = np.linspace(-20, 30, 4)

edges_per_axis = [edges_fx, edges_fy, edges_fz, edges_mx, edges_my, edges_mz]
axes_for_ranges = [0, 1, 2, 3, 4, 5]

pct = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
min_tol = np.array([1, 1, 1, 0.5, 0.5, 0.5])




# --- 6 graphs: one per channel (hist overlay) ---
axis_names = ["Ch0", "Ch1", "Ch2", "Ch3", "Ch4", "Ch5"]  # renomme si tu veux



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

protocols = generate_random_protocols(groups, n_protocols= 1, seed=0)
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



print_best_protocol_trials(best, trials, max_protocols=10, show_file=False)


print(count_all_protocols(groups))
print(generate_random_protocols(groups, n_protocols=5))
print(f"Total groups: {len(groups)}")
print(f"Total trials: {len(trials)}")
for i, idxs in enumerate(groups.values()):
    print(f"Group {i}: {len(idxs)} trials")



