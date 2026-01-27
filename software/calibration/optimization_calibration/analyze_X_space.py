from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from software.calibration.cross_validation import load_force_trials
from software.calibration.calculate_base_AccBias import load_fixed_imu_params
from software.calibration.optimization_calibration.features import build_XY


# ----------------------------
# Helpers: PCA + angles
# ----------------------------

def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA 2D via SVD on centered X.
    Returns:
      Z: (N,2) projected points
      components: (2,6) principal directions
      explained_var_ratio: (2,) proportion of variance explained
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD: Xc = U S Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # eigenvalues of covariance ~ (S^2)/(N-1)
    N = X.shape[0]
    eig = (S**2) / max(N - 1, 1)
    total = float(np.sum(eig)) if np.sum(eig) > 0 else 1.0
    explained = eig / total

    components = Vt[:2, :]          # (2,6)
    Z = Xc @ components.T           # (N,2)
    return Z, components, explained[:2]


def pairwise_angle_stats(X: np.ndarray, n_pairs: int = 20000, seed: int = 0) -> dict:
    """
    Sample random pairs and compute angles between vectors in degrees.
    Useful to see if data directions are diverse or all aligned.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N < 2:
        return {"n_pairs": 0, "mean_deg": np.nan, "p10_deg": np.nan, "p50_deg": np.nan, "p90_deg": np.nan}

    # normalize rows
    norms = np.linalg.norm(X, axis=1)
    valid = norms > 1e-12
    Xn = X[valid] / norms[valid, None]
    M = Xn.shape[0]
    if M < 2:
        return {"n_pairs": 0, "mean_deg": np.nan, "p10_deg": np.nan, "p50_deg": np.nan, "p90_deg": np.nan}

    n_pairs = int(min(n_pairs, M * (M - 1) // 2))
    idx1 = rng.integers(0, M, size=n_pairs)
    idx2 = rng.integers(0, M, size=n_pairs)

    # avoid identical pairs
    same = idx1 == idx2
    while np.any(same):
        idx2[same] = rng.integers(0, M, size=int(np.sum(same)))
        same = idx1 == idx2

    dots = np.sum(Xn[idx1] * Xn[idx2], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))

    return {
        "n_pairs": int(n_pairs),
        "mean_deg": float(np.mean(ang)),
        "p10_deg": float(np.percentile(ang, 10)),
        "p50_deg": float(np.percentile(ang, 50)),
        "p90_deg": float(np.percentile(ang, 90)),
    }


# ----------------------------
# B) Clustering
# ----------------------------

def kmeans_numpy(X: np.ndarray, k: int, n_iter: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple K-means (no sklearn). Returns (labels, centers).
    X: (N,d)
    labels: (N,)
    centers: (k,d)
    """
    rng = np.random.default_rng(seed)
    N, d = X.shape

    # init centers from random points
    centers = X[rng.choice(N, size=k, replace=False)].copy()

    for _ in range(n_iter):
        # assign
        # distances (N,k)
        dist2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dist2, axis=1)

        # update
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centers[j] = X[mask].mean(axis=0)
            else:
                # empty cluster -> reinit
                new_centers[j] = X[rng.integers(0, N)]
        # convergence check
        if np.linalg.norm(new_centers - centers) < 1e-9:
            centers = new_centers
            break
        centers = new_centers

    # final labels
    dist2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(dist2, axis=1)
    return labels, centers


def zone_bins_norm_direction(X: np.ndarray, n_norm_bins: int = 3) -> np.ndarray:
    """
    Simple zone definition:
      - bin by ||X|| (quantiles)
      - direction = axis with max |component| in X (dominant channel)
    Returns zone_id for each sample.
    """
    norms = np.linalg.norm(X, axis=1)
    qs = np.quantile(norms, np.linspace(0, 1, n_norm_bins + 1))
    # digitize into bins [0..n_norm_bins-1]
    norm_bin = np.clip(np.digitize(norms, qs[1:-1], right=False), 0, n_norm_bins - 1)

    dom = np.argmax(np.abs(X), axis=1)  # 0..5
    zone_id = norm_bin * 6 + dom        # unique id
    return zone_id


# ----------------------------
# Main
# ----------------------------

def main():
    # ---- Load your data (adapt paths) ----
    path = Path(__file__).resolve().parent.parent
    imu_dir = path / "E1_E2"
    packages_root = path / "package_trials_good"

    trials = load_force_trials(packages_root)
    base, acc_bias, imu_info = load_fixed_imu_params(imu_dir)

    # ---- Build X (Channels) ----
    X, Y, y_scale = build_XY(trials, acc_bias, base)  # X: (N,6)

    X = np.asarray(X, float)

    print("X shape:", X.shape)

    # =========================
    # A) Analyse l'espace X
    # =========================
    norms = np.linalg.norm(X, axis=1)
    print("||X|| stats:",
          "min", float(np.min(norms)),
          "p10", float(np.percentile(norms, 10)),
          "median", float(np.percentile(norms, 50)),
          "p90", float(np.percentile(norms, 90)),
          "max", float(np.max(norms)))

    ang_stats = pairwise_angle_stats(X, n_pairs=20000, seed=0)
    print("Angle stats (deg):", ang_stats)

    # PCA 2D
    Z, comps, exp = pca_2d(X)
    print("PCA explained variance ratio (PC1, PC2):", exp)

    # Plot norms histogram
    plt.figure()
    plt.hist(norms, bins=40)
    plt.title("Distribution des normes ||X_i|| (Channels)")
    plt.xlabel("||X||")
    plt.ylabel("count")

    # Plot PCA scatter (no colors yet)
    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], alpha=0.6)
    plt.title(f"PCA 2D de X (explained PC1={exp[0]:.2f}, PC2={exp[1]:.2f})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # =========================
    # B) Zones / clusters
    # =========================

    # 1) K-means on PCA space (more stable than raw 6D sometimes)
    k = 6
    labels_km, centers = kmeans_numpy(Z, k=k, n_iter=60, seed=0)

    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c=labels_km, alpha=0.7)
    plt.title(f"Zones par K-means (k={k}) sur PCA(X)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # 2) Simple bins (norm + dominant direction)
    zone_id = zone_bins_norm_direction(X, n_norm_bins=3)

    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c=zone_id, alpha=0.7)
    plt.title("Zones simples: bins(||X||) + direction dominante")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # Print zone counts
    unique_z, counts = np.unique(zone_id, return_counts=True)
    print("Zone bins counts (zone_id -> count), top 15:")
    order = np.argsort(-counts)
    for i in order[:15]:
        print(int(unique_z[i]), "->", int(counts[i]))

    plt.show()


if __name__ == "__main__":
    main()
