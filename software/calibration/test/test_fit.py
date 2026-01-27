from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from software.calibration.cross_validation import (
    train_test_split_trials,
    load_force_trials,
)
from software.calibration.calculate_base_AccBias import load_fixed_imu_params
from software.calibration.fit_A_train import build_FMs_Channels

from software.calibration.optimization_calibration import FitConfig, MonteCarloConfig, InfluenceConfig
from software.calibration.optimization_calibration.features import build_XY
from software.calibration.optimization_calibration.fit import fit
from software.calibration.optimization_calibration.influence import influence_analytic

from software.calibration.optimization_calibration.metrics import (
    RMSE_total, R2_total, RMSE_per_axis, R2_per_axis
)
from software.calibration.optimization_calibration.montecarlo import bootstrap_A
from software.calibration.optimization_calibration.plot import (
    plot_pred_vs_true,
    plot_residuals,
)
from software.calibration.optimization_calibration.report import rank_trials, export_csv, export_json

# ----------------------------
# Paths / data loading
# ----------------------------
path = Path(__file__).resolve().parent.parent          # ../calibration
imu_dir = path / "Base_AccBias_nicolas"
packages_root = path / "trials_forces_nicolas"          # attention au nom exact du dossier
trials = load_force_trials(packages_root)

base, acc_bias, imu_info = load_fixed_imu_params(imu_dir)

train, test = train_test_split_trials(
    trials, test_ratio=0.2, seed=42, stratify_by_mass=True
)

# ----------------------------
# Build matrices
#   Channels = "X", FM = "Y"
# ----------------------------
X_train, Y_train, y_scale = build_XY(train, acc_bias, base)
X_test, Y_test, y_scale   = build_XY(test,  acc_bias, base)

# ----------------------------
# Fit
# ----------------------------
fit_configuration = FitConfig(method="ols", intercept=False)
results = fit(X_train, Y_train, fit_configuration)   # X=Ch, Y=FM
print(results)

# ----------------------------
# Predict on test
# Convention: Y_pred = X @ A.T (+ b0)
# ----------------------------
A = results.A
b0 = getattr(results, "b0", None)  # au cas où ton FitResult n'a pas b0

Y_pred = (X_test @ A.T)
if b0 is not None:
    Y_pred = Y_pred + b0[None, :]

# ----------------------------
# Metrics
# ----------------------------
rmse_total = RMSE_total(Y_test, Y_pred)
rmse_per_axis = RMSE_per_axis(Y_test, Y_pred)

r2_total = R2_total(Y_test, Y_pred)
r2_per_axis = R2_per_axis(Y_test, Y_pred)

print("RMSE total:", rmse_total)
print("RMSE axes :", rmse_per_axis)
print("R2 total  :", r2_total)
print("R2 axes   :", r2_per_axis)

# ----------------------------
# Plots
# ----------------------------
plot_pred_vs_true(Y_test, Y_pred, title_prefix="Calibration — Test", split_forces_moments=True)
plot_residuals(Y_test, Y_pred, title_prefix="Calibration — Test", split_forces_moments=True)

plt.show()


# ----------------------------
# Bootstrap
# ----------------------------

mc_cfg = MonteCarloConfig(n_draws=200, frac=0.8)
res = bootstrap_A(train, acc_bias, base, builder=build_FMs_Channels, fit_configuration=fit_configuration, montecarlo_configuration=mc_cfg)

print("A_mean:\n", res.A_mean)
print("A_std (mean over coefficients):", float(np.mean(res.A_std)))
print("A_std (max coefficients):", float(np.max(res.A_std)))

influence_config = InfluenceConfig()
influence = influence_analytic(
    X = X_train,
    Y = Y_train,
    A = res.A_mean,
    b0 = None,
    cfg = influence_config,
)

rows = rank_trials(influence, train)

out_dir = Path(__file__).resolve().parent
export_csv(rows, str(out_dir / "trial_ranking.csv"))
export_json(rows, str(out_dir / "trial_ranking.json"))

print("Top 10 trials:")
for r in rows[:10]:
    print(f"#{r.idx} score={r.score:.3f} loo_rmse={r.loo_rmse:.3g} cook={r.cook:.3g} deltaA={r.deltaA_Frobenius:.3g} file={r.meta.get('__file__')}")