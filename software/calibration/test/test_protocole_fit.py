from pathlib import Path
from matplotlib import pyplot as plt

from software.calibration.optimization_calibration.features import build_XY
from software.calibration.optimization_calibration.fit import fit
from software.calibration.optimization_calibration.plot import plot_pred_vs_true, plot_residuals
from software.calibration.optimization_calibration.protocole import recommend_protocol, select_trials
from software.calibration.optimization_calibration import FitConfig, MonteCarloConfig
from software.calibration.cross_validation import (
    train_test_split_trials,
    load_force_trials,
)
from software.calibration.calculate_base_AccBias import load_fixed_imu_params

fit_cfg = FitConfig(method="ols", intercept=False)
mc_cfg = MonteCarloConfig(n_draws=300, frac=0.8, seed=42)
path = Path(__file__).resolve().parent.parent          # ../calibration
imu_dir = path / "E1_E2"
packages_root = path / "package_trials_good"          # attention au nom exact du dossier
trials = load_force_trials(packages_root)

base, acc_bias, imu_info = load_fixed_imu_params(imu_dir)

best, candidates = recommend_protocol(trials, acc_bias, base, fit_cfg, mc_cfg, out_dir=Path(__file__).resolve().parent)

spec_by_name = {s.name: s for s in candidates}
print(f"Spec_by_name keys: {list(spec_by_name.keys())}")
print("Top protocols:")
for score, ev in best:
    print(score, ev.name, ev.n_total, ev.rmse_total, ev.A_cv_mean)

# ----------------------------
# Plots on top protocols
# ----------------------------
for rank, (score, ev) in enumerate(best, start=1):
    spec = spec_by_name[ev.name]
    chosen = select_trials(trials, spec, seed=42)

    train_trials, test_trials = train_test_split_trials(
        chosen, test_ratio=0.2, seed=42, stratify_by_mass=True
    )

    # If your build_XY supports normalize_y, keep it consistent with fit_cfg.normalize_y
    X_train, Y_train, _ = build_XY(train_trials, acc_bias, base)
    X_test,  Y_test,  _ = build_XY(test_trials,  acc_bias, base)

    fr = fit(X_train, Y_train, fit_cfg)

    Y_pred = X_test @ fr.A.T
    if getattr(fr, "b0", None) is not None:
        Y_pred = Y_pred + fr.b0[None, :]

    title = f"#{rank} {ev.name} | score={score:.3f} | n={len(chosen)} | rmse={ev.rmse_total:.3f}"
    plot_pred_vs_true(Y_test, Y_pred, title_prefix=title, split_forces_moments=True)
    plot_residuals(Y_test, Y_pred, title_prefix=title, split_forces_moments=True)

plt.show()