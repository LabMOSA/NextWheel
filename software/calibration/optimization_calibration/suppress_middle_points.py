from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from choice_groups import build_choice_groups_by_zone, generate_random_protocols, count_all_protocols
from software.calibration.calculate_base_AccBias import load_fixed_imu_params
from software.calibration.cross_validation import load_force_trials
from features import build_XY
from outliers import find_removed_trials
from plot import plot_pred_vs_true, plot_residuals
from typing import Sequence

from software.calibration.optimization_calibration.metrics import (
    rmse_total as _rmse_total,
    r2_total as _r2_total,
    rmse_per_axis as _rmse_per_axis,
    r2_per_axis as _r2_per_axis,
)

AXES_6 = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


def analyze_conditioning(matrix: np.ndarray) -> dict[str, float | np.ndarray]:
    """Analyze the conditioning of a matrix."""
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    max_singular = np.max(S)
    min_singular = np.min(S)
    condition_number = max_singular / min_singular if min_singular > 0 else np.inf
    return {
        "max_singular_value": max_singular,
        "min_singular_value": min_singular,
        "singular_values": S,
        "condition_number": condition_number,
    }


def suppress_middle_points(
        trials: Sequence[dict],
        acc_bias: np.ndarray,
        base: np.ndarray,
        edges_per_axis: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split points into "extreme" and "middle" sets based on per-axis bin edges.

    A point is considered "extreme" on an axis if it falls in the first or last bin
    defined by the edges for that axis. Points that are extreme on at least one axis
    are kept; the rest are considered "middle" points.

    Parameters
    ----------
    trials : Sequence[dict]
        List of force trial data.
    acc_bias : np.ndarray
        Accelerometer bias.
    base : np.ndarray
        Base transformation matrix.
    edges_per_axis : list[np.ndarray] | None, optional
        List of bin edges for each axis
        (e.g., [edges_fx, edges_fy, edges_fz, edges_mx, edges_my, edges_mz]).
        If None, uses default ranges:
        - Fx: [-100, 120] with 8 bins
        - Fy: [-100, 120] with 8 bins
        - Fz: [-100, 100] with 8 bins
        - Mx: [-50,  50] with 3 bins
        - My: [-50,  50] with 3 bins
        - Mz: [-50,  50] with 3 bins

    Returns
    -------
    X_extreme : np.ndarray
        Feature matrix for extreme points.
    Y_extreme : np.ndarray
        Target matrix for extreme points.
    X_middle : np.ndarray
        Feature matrix for middle (excluded) points.
    Y_middle : np.ndarray
        Target matrix for middle (excluded) points.
    is_extreme : np.ndarray
        Boolean mask of shape (n_samples,); True = extreme point.
    """
    X, Y, _ = build_XY(trials, acc_bias, base)

    if edges_per_axis is None:
        edges_fx = np.linspace(-100, 120, 9)
        edges_fy = np.linspace(-100, 120, 9)
        edges_fz = np.linspace(-100, 100, 9)
        edges_mx = np.linspace(-50, 50, 4)
        edges_my = np.linspace(-50, 50, 4)
        edges_mz = np.linspace(-50, 50, 4)
        edges_per_axis = [edges_fx, edges_fy, edges_fz, edges_mx, edges_my, edges_mz]

    is_extreme = np.zeros(Y.shape[0], dtype=bool)

    for axis in range(min(Y.shape[1], len(edges_per_axis))):
        edges = np.asarray(edges_per_axis[axis])
        y_vals = Y[:, axis]
        in_first_bin = y_vals < edges[1]
        in_last_bin = y_vals >= edges[-2]
        is_extreme |= (in_first_bin | in_last_bin)

    X_extreme = X[is_extreme]
    Y_extreme = Y[is_extreme]
    X_middle = X[~is_extreme]
    Y_middle = Y[~is_extreme]

    return X_extreme, Y_extreme, X_middle, Y_middle, is_extreme


def _print_metrics_block(
        label: str,
        Y_true: np.ndarray,
        Y_pred: np.ndarray,
) -> None:
    """Print a formatted metrics block (RMSE + R²) for a given split."""
    rmse_tot = _rmse_total(Y_true, Y_pred)
    r2_tot = _r2_total(Y_true, Y_pred)
    rmse_axes = _rmse_per_axis(Y_true, Y_pred)
    r2_axes = _r2_per_axis(Y_true, Y_pred)

    print(f"\n{'=' * 60}")
    print(f"  {label}  (n={Y_true.shape[0]})")
    print(f"{'=' * 60}")
    print(f"  RMSE total : {rmse_tot:.6f}")
    print(f"  R²   total : {r2_tot:.6f}")
    print()
    print(f"  {'Axis':<6}  {'RMSE':>12}  {'R²':>12}")
    print(f"  {'-' * 6}  {'-' * 12}  {'-' * 12}")
    for name, rmse_val, r2_val in zip(AXES_6, rmse_axes, r2_axes):
        print(f"  {name:<6}  {rmse_val:>12.6f}  {r2_val:>12.6f}")

def monte_carlo_cross_validation(
        X_extreme: np.ndarray,
        Y_extreme: np.ndarray,
        X_middle: np.ndarray,
        Y_middle: np.ndarray,
        n_draws: int = 300,
        f_train: float = 0.80,
) -> dict:
    """
    Monte Carlo cross-validation on extreme points, with generalisation
    evaluation on middle points at each draw.

    Parameters
    ----------
    X_extreme : np.ndarray
        Feature matrix for extreme points.
    Y_extreme : np.ndarray
        Target matrix for extreme points.
    X_middle : np.ndarray
        Feature matrix for middle points.
    Y_middle : np.ndarray
        Target matrix for middle points.
    n_draws : int
        Number of Monte Carlo draws (default 300).
    f_train : float
        Fraction of extreme points used for training (default 0.80).

    Returns
    -------
    dict
        Aggregated metrics over all draws:
        - rmse_test_mean / std       : RMSE on the 20% extreme test set
        - r2_test_mean / std         : R² on the 20% extreme test set
        - rmse_middle_mean / std     : RMSE on middle points (generalisation)
        - r2_middle_mean / std       : R² on middle points (generalisation)
        - A_cv_mean                  : mean coefficient of variation of Â
    """
    n = X_extreme.shape[0]
    n_train = int(np.floor(f_train * n))

    rmse_test_list = []
    r2_test_list = []
    rmse_middle_list = []
    r2_middle_list = []
    A_list = []

    for _ in range(n_draws):
        # 1. Draw 80% of extreme points for training
        idx = np.random.permutation(n)
        idx_train = idx[:n_train]
        idx_test = idx[n_train:]

        X_tr, Y_tr = X_extreme[idx_train], Y_extreme[idx_train]
        X_te, Y_te = X_extreme[idx_test],  Y_extreme[idx_test]

        # 2. Estimate Â by least squares
        A_hat, _, _, _ = np.linalg.lstsq(X_tr, Y_tr, rcond=None)

        # 3. Evaluate on the remaining 20% extreme points
        Y_pred_te = X_te @ A_hat
        rmse_test_list.append(_rmse_total(Y_te, Y_pred_te))
        r2_test_list.append(_r2_total(Y_te, Y_pred_te))

        # 4. Evaluate generalisation on middle points
        if X_middle.shape[0] > 0:
            Y_pred_mid = X_middle @ A_hat
            rmse_middle_list.append(_rmse_total(Y_middle, Y_pred_mid))
            r2_middle_list.append(_r2_total(Y_middle, Y_pred_mid))

        A_list.append(A_hat)

    # 5. Stability of Â : coefficient of variation
    A_stack = np.stack(A_list, axis=0)        # (n_draws, p, 6)
    A_mean  = np.mean(A_stack, axis=0)
    A_std   = np.std(A_stack,  axis=0)
    mask    = np.abs(A_mean) > 1e-12
    cv      = np.zeros_like(A_std)
    cv[mask] = A_std[mask] / np.abs(A_mean[mask])
    A_cv_mean = float(np.mean(cv[mask])) if np.any(mask) else float("nan")

    return {
        "rmse_test_mean":   float(np.mean(rmse_test_list)),
        "rmse_test_std":    float(np.std(rmse_test_list)),
        "r2_test_mean":     float(np.mean(r2_test_list)),
        "r2_test_std":      float(np.std(r2_test_list)),
        "rmse_middle_mean": float(np.mean(rmse_middle_list)) if rmse_middle_list else float("nan"),
        "rmse_middle_std":  float(np.std(rmse_middle_list))  if rmse_middle_list else float("nan"),
        "r2_middle_mean":   float(np.mean(r2_middle_list))   if r2_middle_list   else float("nan"),
        "r2_middle_std":    float(np.std(r2_middle_list))    if r2_middle_list   else float("nan"),
        "A_cv_mean":        A_cv_mean,
    }

def monte_carlo_cross_validation_B(
        trials_extreme: list[dict],
        X_extreme: np.ndarray,
        Y_extreme: np.ndarray,
        X_middle: np.ndarray,
        Y_middle: np.ndarray,
        edges_per_axis: list[np.ndarray],
        n_draws: int = 300,
) -> dict:
    """
    Monte Carlo cross-validation for Protocol B.

    At each draw, one trial per zone is selected (combination of angular
    positions), Â is estimated on this subset, and generalisation is
    evaluated on middle points.

    Parameters
    ----------
    trials_extreme : list[dict]
        List of extreme trial dicts.
    X_extreme : np.ndarray
        Feature matrix for extreme points (n_extreme, n_features).
    Y_extreme : np.ndarray
        Target matrix for extreme points (n_extreme, 6).
    X_middle : np.ndarray
        Feature matrix for middle points.
    Y_middle : np.ndarray
        Target matrix for middle points.
    edges_per_axis : list[np.ndarray]
        Bin edges per axis, used to define zones.
    n_draws : int
        Number of Monte Carlo draws (default 300).

    Returns
    -------
    dict
        Aggregated metrics over all draws.
    """
    # 1. Build zone groups on extreme points
    groups = build_choice_groups_by_zone(
        trials_extreme,
        X=X_extreme,
        Y=Y_extreme,
        edges_per_axis=edges_per_axis,
    )

    print(f"  Nombre de zones : {len(groups)}")
    print(f"  Nombre total de protocoles possibles : {count_all_protocols(groups)}")

    # 2. Generate n_draws random protocols (one trial per zone each)
    protocols = generate_random_protocols(groups, n_protocols=n_draws, seed=0)

    rmse_middle_list = []
    r2_middle_list = []
    A_list = []
    skipped = 0

    for proto_indices in protocols:
        if len(proto_indices) < 6:
            skipped += 1
            continue

        X_tr = X_extreme[proto_indices]
        Y_tr = Y_extreme[proto_indices]

        # 3. Estimate Â by least squares
        A_hat, _, _, _ = np.linalg.lstsq(X_tr, Y_tr, rcond=None)

        # 4. Evaluate generalisation on middle points
        if X_middle.shape[0] > 0:
            Y_pred_mid = X_middle @ A_hat
            rmse_middle_list.append(_rmse_total(Y_middle, Y_pred_mid))
            r2_middle_list.append(_r2_total(Y_middle, Y_pred_mid))

        A_list.append(A_hat)

    if skipped > 0:
        print(f"  [WARN] {skipped} protocoles écartés (< 6 essais)")

    # 5. Stability of Â
    A_stack = np.stack(A_list, axis=0)
    A_mean  = np.mean(A_stack, axis=0)
    A_std   = np.std(A_stack,  axis=0)
    mask    = np.abs(A_mean) > 1e-12
    cv      = np.zeros_like(A_std)
    cv[mask] = A_std[mask] / np.abs(A_mean[mask])
    A_cv_mean = float(np.mean(cv[mask])) if np.any(mask) else float("nan")

    return {
        "rmse_middle_mean": float(np.mean(rmse_middle_list)) if rmse_middle_list else float("nan"),
        "rmse_middle_std": float(np.std(rmse_middle_list)) if rmse_middle_list else float("nan"),
        "r2_middle_mean": float(np.mean(r2_middle_list)) if r2_middle_list else float("nan"),
        "r2_middle_std": float(np.std(r2_middle_list)) if r2_middle_list else float("nan"),
        "A_cv_mean": A_cv_mean,
        "n_valid_draws": len(A_list),
        "rmse_middle_list": rmse_middle_list,  # ajout
        "r2_middle_list": r2_middle_list,  # ajout
    }

def print_mc_results(results: dict) -> None:
    """Print a formatted summary of Monte Carlo cross-validation results."""
    print(f"\n{'=' * 60}")
    print("  Monte Carlo Cross-Validation Results")
    print(f"{'=' * 60}")
    print(f"  RMSE  (20% extreme test) : "
          f"{results['rmse_test_mean']:.4f} ± {results['rmse_test_std']:.4f}")
    print(f"  R²    (20% extreme test) : "
          f"{results['r2_test_mean']:.4f} ± {results['r2_test_std']:.4f}")
    print(f"\n  RMSE  (middle generalisation) : "
          f"{results['rmse_middle_mean']:.4f} ± {results['rmse_middle_std']:.4f}")
    print(f"  R²    (middle generalisation) : "
          f"{results['r2_middle_mean']:.4f} ± {results['r2_middle_std']:.4f}")
    print(f"\n  CV(Â) mean : {results['A_cv_mean']:.4f}")

def print_mc_results_B(results: dict) -> None:
    """Print a formatted summary of Protocol B Monte Carlo results."""
    print(f"\n{'=' * 60}")
    print("  Monte Carlo Cross-Validation Results — Protocol B")
    print(f"{'=' * 60}")
    print(f"  Tirages valides : {results['n_valid_draws']}")
    print(f"\n  RMSE  (généralisation points médians) : "
          f"{results['rmse_middle_mean']:.4f} ± {results['rmse_middle_std']:.4f}")
    print(f"  R²    (généralisation points médians) : "
          f"{results['r2_middle_mean']:.4f} ± {results['r2_middle_std']:.4f}")
    print(f"\n  CV(Â) mean : {results['A_cv_mean']:.4f}")

def plot_mc_distribution_B(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Distribution RMSE
    axes[0].hist(results["rmse_middle_list"], bins=30, edgecolor="black")
    axes[0].axvline(results["rmse_middle_mean"], color="red",
                    linestyle="--", label=f"Moyenne = {results['rmse_middle_mean']:.3f}")
    axes[0].set_xlabel("RMSE (points médians)")
    axes[0].set_ylabel("Nombre de protocoles")
    axes[0].set_title("Distribution du RMSE sur 300 tirages")
    axes[0].legend()

    # Distribution R²
    axes[1].hist(results["r2_middle_list"], bins=30, edgecolor="black")
    axes[1].axvline(results["r2_middle_mean"], color="red",
                    linestyle="--", label=f"Moyenne = {results['r2_middle_mean']:.3f}")
    axes[1].set_xlabel("R² (points médians)")
    axes[1].set_ylabel("Nombre de protocoles")
    axes[1].set_title("Distribution du R² sur 300 tirages")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    path = Path(__file__).resolve().parent.parent
    imu_dir = path / "E1_E2"
    packages_root = path / "trials_organized"

    trials = load_force_trials(packages_root)
    base, acc_bias, imu_info = load_fixed_imu_params(imu_dir)

    # Remove outliers first (same pipeline as before)
    removed_trials, removed, trials_f, y_true_f, y_pred_f = find_removed_trials()

    # ------------------------------------------------------------------
    # 2. Split into extreme / middle
    # ------------------------------------------------------------------
    X_extreme, Y_extreme, X_middle, Y_middle, is_extreme = suppress_middle_points(
        trials_f, acc_bias, base
    )

    print(f"\nTotal trials after outlier removal : {len(trials_f)}")
    print(f"  Extreme points (kept for calib)  : {X_extreme.shape[0]}"
          f"  ({100 * X_extreme.shape[0] / len(trials_f):.1f} %)")
    print(f"  Middle  points (excluded)        : {X_middle.shape[0]}"
          f"  ({100 * X_middle.shape[0] / len(trials_f):.1f} %)")

    # ------------------------------------------------------------------
    # 3. Monte Carlo cross-validation
    # ------------------------------------------------------------------
    mc_results = monte_carlo_cross_validation(
        X_extreme, Y_extreme,
        X_middle, Y_middle,
        n_draws=300,
        f_train=0.80,
    )


    # ------------------------------------------------------------------
    # 3. Monte Carlo cross-validation Protocol B
    # ------------------------------------------------------------------
    edges_fx = np.linspace(-100, 120, 9)
    edges_fy = np.linspace(-100, 120, 9)
    edges_fz = np.linspace(-100, 100, 9)
    edges_mx = np.linspace(-50, 50, 4)
    edges_my = np.linspace(-50, 50, 4)
    edges_mz = np.linspace(-50, 50, 4)
    edges_per_axis = [edges_fx, edges_fy, edges_fz, edges_mx, edges_my, edges_mz]

    # Récupérer les trials extrêmes (mask is_extreme sur trials_f)
    trials_extreme = [t for t, ext in zip(trials_f, is_extreme) if ext]

    print(trials_extreme[0].get("__file__", None))
    print(trials_extreme[40].get("__file__", None))

    from software.calibration.optimization_calibration.recommand_protocol import print_trials_table

    print_trials_table(trials_extreme)
    mc_results_B = monte_carlo_cross_validation_B(
        trials_extreme=trials_extreme,
        X_extreme=X_extreme,
        Y_extreme=Y_extreme,
        X_middle=X_middle,
        Y_middle=Y_middle,
        edges_per_axis=edges_per_axis,
        n_draws=300,
    )
    print_mc_results_B(mc_results_B)
    plot_mc_distribution_B(mc_results_B)
    print_mc_results(mc_results)
    # ------------------------------------------------------------------
    # 3. Load calibration matrix
    # ------------------------------------------------------------------
    ROOT = Path.cwd().parent
    z = np.load(ROOT / "matrix_A_offset.npz")
    A = z["A"]
    y_scale = z["y_scale"]


    # ------------------------------------------------------------------
    # 4. Predict on both splits (denormalised)
    # ------------------------------------------------------------------
    def predict(X: np.ndarray) -> np.ndarray:
        return (X @ A.T) * y_scale[None, :]


    Y_pred_extreme = predict(X_extreme)
    Y_pred_middle = predict(X_middle)

    # ------------------------------------------------------------------
    # 5. Metrics — extreme points (calibration set)
    # ------------------------------------------------------------------
    _print_metrics_block("EXTREME points  (calibration set)", Y_extreme, Y_pred_extreme)

    # ------------------------------------------------------------------
    # 6. Metrics — middle points (generalisation test)
    # ------------------------------------------------------------------
    if X_middle.shape[0] > 0:
        _print_metrics_block("MIDDLE  points  (generalisation test)", Y_middle, Y_pred_middle)
    else:
        print("\n[INFO] No middle points found — nothing to evaluate on excluded set.")


    # ------------------------------------------------------------------
    # 7. Plots — extreme points
    # ------------------------------------------------------------------
    print("\n--- Plots: EXTREME points ---")
    plot_pred_vs_true(
        Y_extreme, Y_pred_extreme,
        title_prefix="Extreme (calib)", split_forces_moments=True,
    )
    plot_residuals(
        Y_extreme, Y_pred_extreme,
        title_prefix="Extreme (calib)", split_forces_moments=True,
    )



    # ------------------------------------------------------------------
    # 8. Plots — middle points
    # ------------------------------------------------------------------
    if X_middle.shape[0] > 0:
        print("--- Plots: MIDDLE points (excluded) ---")
        plot_pred_vs_true(
            Y_middle, Y_pred_middle,
            title_prefix="Middle (generalisation)", split_forces_moments=True,
        )
        plot_residuals(
            Y_middle, Y_pred_middle,
            title_prefix="Middle (generalisation)", split_forces_moments=True,
        )

    plt.show()