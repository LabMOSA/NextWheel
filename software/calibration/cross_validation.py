from __future__ import annotations
from pathlib import Path
from typing import Union

import numpy as np
import kineticstoolkit as ktk
from collections import Counter


# def load_force_trials(packages_trials: str | Path) -> list[dict]:
#     root = Path(packages_trials)
#     trials_calibration = []
#     for pkg in sorted(root.iterdir()):
#         print(f"Checking package: {pkg.name}")
#         if not pkg.is_dir():
#             continue
#         for f in sorted(pkg.glob("ForcesForCalibrationMatrix*")):
#             d = ktk.load(str(f))
#             d["__file__"] = str(f)
#             d["__package__"] = pkg.name
#             trials_calibration.append(d)
#     if not trials_calibration:
#         raise FileNotFoundError(f"No force trials found under {root}")
#     return trials_calibration

def load_force_trials(packages_trials: Union[str, Path]) -> list[dict]:
    root = Path(packages_trials)
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    trials_calibration: list[dict] = []

    # Cherche récursivement tous les fichiers ForcesForCalibrationMatrix*
    files = sorted(root.rglob("ForcesForCalibrationMatrix*"))

    for f in files:
        if not f.is_file():
            continue

        # "package" = 1er dossier sous root (ex: Protocol_V1)
        try:
            package = f.relative_to(root).parts[0]
        except Exception:
            package = f.parent.name  # fallback


        d = ktk.load(str(f))
        d["__file__"] = str(f)
        d["__package__"] = package
        d["__relpath__"] = str(f.relative_to(root))  # utile pour debug/tri

        trials_calibration.append(d)

    if not trials_calibration:
        raise FileNotFoundError(
            f"No force trials found under {root} (pattern: ForcesForCalibrationMatrix*)"
        )

    return trials_calibration

def train_test_split_trials(trials: list[dict], test_ratio: float = 0.2, seed: int = 0, stratify_by_mass: bool = True):
    rng = np.random.default_rng(seed)

    if not stratify_by_mass:
        idx = np.arange(len(trials))
        rng.shuffle(idx)
        n_test = max(1, int(round(test_ratio * len(trials))))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        return [trials[i] for i in train_idx], [trials[i] for i in test_idx]

    # stratified by Mass
    masses = {}
    for i, t in enumerate(trials):
        m = float(t.get("Mass", np.nan))
        masses.setdefault(m, []).append(i)

    train_idx, test_idx = [], []
    for m, idxs in masses.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_test_m = max(1, int(round(test_ratio * len(idxs))))
        test_idx.extend(idxs[:n_test_m].tolist())
        train_idx.extend(idxs[n_test_m:].tolist())

    return [trials[i] for i in train_idx], [trials[i] for i in test_idx]

def summarize_masses(trials_list):
    masses = [float(t.get("Mass", np.nan)) for t in trials_list]
    return Counter(masses)

def summarize_masses(trials_list):
    masses = [float(t.get("Mass", np.nan)) for t in trials_list]
    return Counter(masses)

def check_split(trials, train, test):
    # 1) tailles
    print("Total:", len(trials), "Train:", len(train), "Test:", len(test))

    # 2) pas de recouvrement (on se base sur __file__ qui est unique)
    train_files = {t["__file__"] for t in train}
    test_files = {t["__file__"] for t in test}
    overlap = train_files.intersection(test_files)
    print("Overlap files:", len(overlap))
    if overlap:
        print("Example overlap:", list(overlap)[:3])

    # 3) couverture des masses
    print("Masses in TRAIN:", summarize_masses(train))
    print("Masses in TEST :", summarize_masses(test))

if __name__ == "__main__":
    packages_root = Path(__file__).resolve().parent / "2_planes_optimization_trials"
    trials = load_force_trials(packages_root)
    for trial in trials:
        print(f"Package: {trial['__package__']}, File: {trial['__file__']}, Mass: {trial['Mass']}, Degree: {trial['Degree']}")

    # Test 1: stratified split
    train, test = train_test_split_trials(trials, test_ratio=0.2, seed=42, stratify_by_mass=True)
    print("\n--- Stratified split ---")
    check_split(trials, train, test)

    # Test 2: non-stratified split
    train2, test2 = train_test_split_trials(trials, test_ratio=0.2, seed=42, stratify_by_mass=False)
    print("\n--- Random split (no stratify) ---")
    check_split(trials, train2, test2)