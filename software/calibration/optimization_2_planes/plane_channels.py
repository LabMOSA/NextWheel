import itertools
from pathlib import Path

import numpy as np

from software.calibration.calculate_base_AccBias import load_fixed_imu_params
from software.calibration.cross_validation import load_force_trials
from software.calibration.optimization_2_planes.features import build_XY
from software.calibration.optimization_2_planes.organize_trials import assign_planes

AXES = {"Fx":0, "Fy":1, "Fz":2, "Mx":3, "My":4, "Mz":5}

def fit_ols(X, Y):
    # X: (n, p), Y: (n, q)
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (p, q)
    return B

def rmse(Y, Yhat):
    e = Y - Yhat
    return float(np.sqrt(np.mean(e**2)))

def best_3_channels_for_plane(X, Y, out_idx, train_frac=0.8, seed=0, score="rmse_total"):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    ntr = int(np.floor(train_frac * n))
    tr = perm[:ntr]
    te = perm[ntr:]

    Yp = Y[:, out_idx]  # sorties du plan
    best = None

    for in_idx in itertools.combinations(range(X.shape[1]), 3):
        Xi_tr = X[tr][:, in_idx]
        Yi_tr = Yp[tr]
        Xi_te = X[te][:, in_idx]
        Yi_te = Yp[te]

        # fit
        B = fit_ols(Xi_tr, Yi_tr)           # (3, 3)
        Yhat = Xi_te @ B                   # (n_test, 3)

        s = rmse(Yi_te, Yhat)              # score
        if (best is None) or (s < best["score"]):
            best = {"in_idx": in_idx, "out_idx": tuple(out_idx), "score": s, "B": B}

    return best

# Exemple d'utilisation:
path = Path(__file__).resolve().parent.parent
imu_dir = path / "E1_E2"
packages_root = path / "package_trials_good"
trials = load_force_trials(packages_root)
base, acc_bias, imu_info = load_fixed_imu_params(imu_dir)

# Split trials
out = assign_planes(trials)
trials_plane_A = out["plane_FxFyMz"]
trials_plane_B = out["plane_MxMyFz"]

X_plane_A, Y_plane_A, _ = build_XY(trials_plane_A, acc_bias, base)
X_plane_B, Y_plane_B, _ = build_XY(trials_plane_B, acc_bias, base)
best_A = best_3_channels_for_plane(X_plane_A, Y_plane_A, out_idx=[AXES["Fx"], AXES["Fy"], AXES["Mz"]], seed=42)
best_B = best_3_channels_for_plane(X_plane_B, Y_plane_B, out_idx=[AXES["Mx"], AXES["My"], AXES["Fz"]], seed=42)

print("Plan A (Fx,Fy,Mz):", best_A["in_idx"], "RMSE:", best_A["score"])
print("Plan B (Mx,My,Fz):", best_B["in_idx"], "RMSE:", best_B["score"])
