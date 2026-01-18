import numpy as np
import kineticstoolkit as ktk
from pathlib import Path
import wheelcalibration as wc
from cross_validation import load_force_trials, train_test_split_trials
from calculate_base_AccBias import load_fixed_imu_params

def build_FMs_Channels(force_trials: list[dict], acc_bias: np.ndarray, base: np.ndarray):
    FMs, Channels = [], []
    for t in force_trials:
        # print(f"\nProcessing trial: {t['__file__']}")
        fm = wc.make_an_estimation_of_forces_moments(t, acc_bias, base)
        # print(f"Trial {t['__file__']} -> Estimated FM: {fm}")
        ch = np.median(t["Analog"]["Force"], axis=0)
        # print(f"Trial {t['__file__']} -> Channel (median Analog Force): {ch}")
        F = np.asarray(t["Analog"]["Force"])
        # print("median:", np.median(F, axis=0))
        # print("std:", F.std(axis=0))
        # print("ptp:", np.ptp(F, axis=0))

        FMs.append(fm)
        Channels.append(ch)
    return np.vstack(FMs), np.vstack(Channels)

def fit_A(force_trials_train: list[dict], acc_bias: np.ndarray, base: np.ndarray) -> np.ndarray:
    FMs, Channels = build_FMs_Channels(force_trials_train, acc_bias, base)
    return wc.calculate_calibration_matrix(FMs, Channels)

def evaluate_A(force_trials_test: list[dict], A: np.ndarray, acc_bias: np.ndarray, base: np.ndarray) -> dict:
    FMs, Channels = build_FMs_Channels(force_trials_test, acc_bias, base)
    FMs_pred = (A @ Channels.T).T
    err = FMs_pred - FMs
    rmse_per_axis = np.sqrt(np.mean(err**2, axis=0))
    rmse_total = float(np.sqrt(np.mean(err**2)))
    return {"rmse_total": rmse_total, "rmse_per_axis": rmse_per_axis, "n_test": len(force_trials_test)}




def quick_trial_info(trial: dict) -> str:
    mass = trial.get("Mass", None)
    deg = trial.get("Degree", None)
    force = np.asarray(trial["Analog"]["Force"])
    return f"Mass={mass}, Degree={deg}, AnalogForceShape={force.shape}"


def main():

    # ---- 1) Paths
    path = Path(__file__).resolve().parent
    imu_dir = path / "E1_E2"
    forces_root = path / "package_trials_good"

    # ---- 2) Load fixed IMU params
    base, acc_bias, imu_info = load_fixed_imu_params(imu_dir)
    print("IMU files used:", imu_info["files_used"])
    print("Base shape:", np.asarray(base).shape)
    print("AccBias shape:", np.asarray(acc_bias).shape)

    # ---- 3) Load force trials
    trials = load_force_trials(forces_root)
    print("\nNumber of force trials loaded:", len(trials))
    print("First trial:", quick_trial_info(trials[0]))
    print("Last  trial:", quick_trial_info(trials[-1]))

    # ---- 4) Split
    train, test = train_test_split_trials(trials, test_ratio=0.2, seed=42, stratify_by_mass=True)
    print("\nSplit sizes -> train:", len(train), "test:", len(test))

    # Sanity: no overlap
    train_files = {t["__file__"] for t in train}
    test_files = {t["__file__"] for t in test}
    overlap = train_files & test_files
    print("Overlap files:", len(overlap))
    assert len(overlap) == 0, "Train/test overlap detected!"

    # ---- 5) Build matrices (train)
    FMs_train, Ch_train = build_FMs_Channels(train, acc_bias, base)
    print("\nTrain matrices shapes:")
    print("FMs_train:", FMs_train.shape)       # (N_train, 6)
    print("Ch_train:", Ch_train.shape)         # (N_train, 6)
    assert FMs_train.shape[0] == len(train)
    assert FMs_train.shape[1] == 6
    assert Ch_train.shape == FMs_train.shape

    # ---- 6) Fit A
    A = fit_A(train, acc_bias, base)
    print("\nA shape:", A.shape)
    assert A.shape == (6, 6), "Expected A to be 6x6"

    # ---- 7) Evaluate
    metrics = evaluate_A(test, A, acc_bias, base)
    print("\n--- METRICS ---")
    print("RMSE total:", metrics["rmse_total"])
    print("RMSE per axis:", metrics["rmse_per_axis"])
    print("n_test:", metrics["n_test"])

    # ---- 8) Quick “train error” (optional, should be <= test usually)
    metrics_train = evaluate_A(train, A, acc_bias, base)
    print("\nTrain RMSE total:", metrics_train["rmse_total"])

    # ---- 9) Basic numeric checks
    assert np.isfinite(metrics["rmse_total"])
    assert np.all(np.isfinite(metrics["rmse_per_axis"]))

    print("\n All tests passed.")


if __name__ == "__main__":
    main()
