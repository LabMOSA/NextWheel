# from __future__ import annotations
#
# from typing import Callable, Any, Sequence
# import numpy as np
#
#
# # def zone_id_from_X_pca_angle(X: np.ndarray, n_bins: int = 6) -> np.ndarray:
# #     """
# #     PCA -> 2D (via SVD), then angular bins.
# #
# #     Returns zone_id in [0..n_bins-1], one per row of X.
# #     Robust to small n: if n < 2 or PCA can't produce 2D, returns all zeros.
# #     """
# #     X = np.asarray(X, float)
# #     n = X.shape[0]
# #     if n == 0:
# #         return np.array([], dtype=int)
# #     if n < 2:
# #         return np.zeros(n, dtype=int)
# #
# #     Xc = X - X.mean(axis=0, keepdims=True)
# #
# #     _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
# #     if Vt.shape[0] < 2:
# #         return np.zeros(n, dtype=int)
# #
# #     Z = Xc @ Vt[:2].T  # (n,2)
# #     if Z.shape[1] < 2:
# #         return np.zeros(n, dtype=int)
# #
# #     ang = np.arctan2(Z[:, 1], Z[:, 0])         # [-pi, pi]
# #     ang01 = (ang + np.pi) / (2.0 * np.pi)      # [0, 1)
# #     zone_id = np.floor(ang01 * n_bins).astype(int)
# #     zone_id = np.clip(zone_id, 0, n_bins - 1)
# #     return zone_id
# #
# #
# # def zone_id_from_Y_ranges(
# #     Y: np.ndarray,
# #     edges_per_axis: Sequence[np.ndarray],
# #     axes: Sequence[int] | None = None,
# # ) -> np.ndarray:
# #     """
# #     Build a composite zone_id from per-axis numeric ranges (bins) on Y.
# #
# #     edges_per_axis: list of edges arrays, one per axis in `axes`
# #       e.g. [np.linspace(-40,40,9), np.linspace(-40,40,9), np.linspace(0,200,9)]
# #     axes: which columns of Y to use. If None, uses first len(edges_per_axis) axes.
# #
# #     Returns zone_id in [0..prod(n_bins_axis)-1], one per row of Y.
# #     """
# #     Y = np.asarray(Y, float)
# #     n = Y.shape[0]
# #     if n == 0:
# #         return np.array([], dtype=int)
# #
# #     if axes is None:
# #         axes = list(range(len(edges_per_axis)))
# #     if len(axes) != len(edges_per_axis):
# #         raise ValueError("axes and edges_per_axis must have the same length")
# #
# #     # Per-axis bin ids
# #     per_axis_bins: list[np.ndarray] = []
# #     sizes: list[int] = []
# #     for ax, edges in zip(axes, edges_per_axis):
# #         edges = np.asarray(edges, float)
# #         if edges.ndim != 1 or edges.size < 2:
# #             raise ValueError("Each edges array must be 1D with at least 2 values")
# #
# #         z = np.digitize(Y[:, ax], edges, right=False) - 1  # -> [0..len(edges)-2] ideally
# #         z = np.clip(z, 0, edges.size - 2)
# #         per_axis_bins.append(z.astype(int))
# #         sizes.append(int(edges.size - 1))  # number of bins
# #
# #     # Mixed-radix encoding: zone = z0 + z1*s0 + z2*s0*s1 + ...
# #     zone = np.zeros(n, dtype=int)
# #     mult = 1
# #     for z, s in zip(per_axis_bins, sizes):
# #         zone += z * mult
# #         mult *= s
# #
# #     return zone
# #
# #
# # def cap_trials_per_zone_mass_pca_and_ranges(
# #     trials: list[dict],
# #     X: np.ndarray,
# #     Y: np.ndarray,
# #     k_per_key: int,
# #     mass_fn: Callable[[dict], Any],
# #     seed: int = 0,
# #     # PCA zoning (on X):
# #     pca_bins: int = 6,
# #     # Range zoning (on Y):
# #     edges_per_axis: Sequence[np.ndarray] | None = None,
# #     axes_for_ranges: Sequence[int] | None = None,
# #     # Optional: if you already computed zone ids, pass them
# #     zone_pca: np.ndarray | None = None,
# #     zone_ranges: np.ndarray | None = None,
# # ) -> list[dict]:
# #     """
# #     Keep at most k_per_key trials per (mass, PCA_zone, RANGE_zone).
# #
# #     - PCA_zone comes from X via PCA->2D->angular bins (captures "sensor-space geometry").
# #     - RANGE_zone comes from Y via per-axis bins (captures "physical excitation coverage").
# #     - Combining both avoids selecting many redundant points and also forces coverage of desired ranges.
# #
# #     Parameters
# #     ----------
# #     trials : list[dict]
# #         Trials list.
# #     X : np.ndarray
# #         Feature matrix aligned with trials, shape (n_trials, n_features).
# #     Y : np.ndarray
# #         Target matrix aligned with trials, shape (n_trials, n_axes) (e.g., FM).
# #     k_per_key : int
# #         Max number kept per bucket.
# #     mass_fn : Callable
# #         Extracts mass value from a trial (e.g., lambda t: t["Mass"]).
# #     edges_per_axis : Sequence[np.ndarray] | None
# #         Defines the desired ranges/bins on Y. If None, range-zoning is disabled (all in one range zone).
# #     axes_for_ranges : Sequence[int] | None
# #         Which Y axes the edges correspond to. If None, uses first len(edges_per_axis).
# #
# #     Returns
# #     -------
# #     list[dict]
# #         Selected trials.
# #     """
# #     if k_per_key is None:
# #         return trials
# #     k_per_key = int(k_per_key)
# #     if k_per_key <= 0:
# #         return []
# #
# #     X = np.asarray(X, float)
# #     Y = np.asarray(Y, float)
# #     if X.shape[0] != len(trials):
# #         raise ValueError(f"X rows ({X.shape[0]}) must match trials ({len(trials)})")
# #     if Y.shape[0] != len(trials):
# #         raise ValueError(f"Y rows ({Y.shape[0]}) must match trials ({len(trials)})")
# #
# #     rng = np.random.default_rng(seed)
# #
# #     # --- PCA zone on X ---
# #     if zone_pca is None:
# #         zone_pca = zone_id_from_X_pca_angle(X, n_bins=pca_bins)
# #     zone_pca = np.asarray(zone_pca, dtype=int)
# #     if zone_pca.shape[0] != len(trials):
# #         raise ValueError("zone_pca must have shape (n_trials,) and match trials length")
# #
# #     # --- Range zone on Y ---
# #     if edges_per_axis is None:
# #         # single "range zone" if user didn't specify bins
# #         zone_ranges = np.zeros(len(trials), dtype=int)
# #     else:
# #         if zone_ranges is None:
# #             zone_ranges = zone_id_from_Y_ranges(Y, edges_per_axis=edges_per_axis, axes=axes_for_ranges)
# #         zone_ranges = np.asarray(zone_ranges, dtype=int)
# #         if zone_ranges.shape[0] != len(trials):
# #             raise ValueError("zone_ranges must have shape (n_trials,) and match trials length")
# #
# #     # --- Bucket by (mass, pca_zone, range_zone) ---
# #     buckets: dict[tuple[Any, int, int], list[int]] = {}
# #     for i, t in enumerate(trials):
# #         key = (mass_fn(t), int(zone_pca[i]), int(zone_ranges[i]))
# #         buckets.setdefault(key, []).append(i)
# #
# #     kept: list[int] = []
# #     for idxs in buckets.values():
# #         idxs = np.asarray(idxs, dtype=int)
# #         rng.shuffle(idxs)
# #         kept.extend(idxs[: min(k_per_key, len(idxs))].tolist())
# #
# #     kept = sorted(set(kept))
# #     return [trials[i] for i in kept]
#
#
#
from __future__ import annotations

from typing import Callable, Any, Sequence
import numpy as np


def zone_id_from_X_pca_angle_radius(X: np.ndarray, n_ang: int = 6, n_rad: int = 3) -> np.ndarray:
    """
    PCA -> 2D (via SVD), then zoning by:
      - angle bins (n_ang)
      - radius bins (n_rad), using quantiles so bins are populated

    Returns a single zone_id in [0 .. n_ang*n_rad - 1] per row of X.

    Robust to small/degenerate inputs: if PCA can't make 2D or n<2, returns all zeros.
    """
    X = np.asarray(X, float)
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n < 2:
        return np.zeros(n, dtype=int)

    Xc = X - X.mean(axis=0, keepdims=True)

    # PCA via SVD
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    if Vt.shape[0] < 2:
        return np.zeros(n, dtype=int)

    Z = Xc @ Vt[:2].T  # (n,2)
    if Z.shape[1] < 2:
        return np.zeros(n, dtype=int)

    # --- angle bin ---
    ang = np.arctan2(Z[:, 1], Z[:, 0])               # [-pi, pi]
    ang01 = (ang + np.pi) / (2.0 * np.pi)            # [0,1)
    z_ang = np.floor(ang01 * n_ang).astype(int)
    z_ang = np.clip(z_ang, 0, n_ang - 1)

    # --- radius bin (quantiles) ---
    r = np.sqrt(Z[:, 0] ** 2 + Z[:, 1] ** 2)

    # If all radii equal, everything in the same radial bin
    if np.allclose(r, r[0]):
        z_rad = np.zeros(n, dtype=int)
    else:
        qs = np.linspace(0.0, 1.0, n_rad + 1)
        edges = np.quantile(r, qs)

        # Remove duplicate edges (can happen if many radii are equal)
        edges = np.unique(edges)
        if edges.size < 2:
            z_rad = np.zeros(n, dtype=int)
        else:
            z_rad = np.digitize(r, edges, right=False) - 1
            z_rad = np.clip(z_rad, 0, edges.size - 2)
            # If edges collapsed, keep within [0..n_rad-1]
            z_rad = np.minimum(z_rad, n_rad - 1).astype(int)

    # Combine bins
    return (z_rad * n_ang + z_ang).astype(int)


def zone_id_from_Y_ranges(
    Y: np.ndarray,
    edges_per_axis: Sequence[np.ndarray],
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Composite zone_id from per-axis numeric ranges (bins) on Y.

    Returns zone_id in [0..prod(n_bins_axis)-1], one per row of Y.
    """
    Y = np.asarray(Y, float)
    n = Y.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    if axes is None:
        axes = list(range(len(edges_per_axis)))
    if len(axes) != len(edges_per_axis):
        raise ValueError("axes and edges_per_axis must have the same length")

    per_axis_bins: list[np.ndarray] = []
    sizes: list[int] = []

    for ax, edges in zip(axes, edges_per_axis):
        edges = np.asarray(edges, float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Each edges array must be 1D with at least 2 values")

        z = np.digitize(Y[:, ax], edges, right=False) - 1
        z = np.clip(z, 0, edges.size - 2)
        per_axis_bins.append(z.astype(int))
        sizes.append(int(edges.size - 1))

    zone = np.zeros(n, dtype=int)
    mult = 1
    for z, s in zip(per_axis_bins, sizes):
        zone += z * mult
        mult *= s

    return zone


# def cap_trials_per_zone_mass_pca_and_ranges(
#     trials: list[dict],
#     X: np.ndarray,
#     Y: np.ndarray,
#     k_per_key: int,
#     mass_fn: Callable[[dict], Any],
#     seed: int = 0,
#     # PCA zoning (on X): angle+radius bins
#     pca_n_ang: int = 6,
#     pca_n_rad: int = 3,
#     # Range zoning (on Y):
#     edges_per_axis: Sequence[np.ndarray] | None = None,
#     axes_for_ranges: Sequence[int] | None = None,
#     # Optional: if you already computed zone ids, pass them
#     zone_pca: np.ndarray | None = None,
#     zone_ranges: np.ndarray | None = None,
# ) -> list[dict]:
#     """
#     Keep at most k_per_key trials per (mass, PCA_zone, RANGE_zone).
#
#     PCA_zone:
#       - computed from X by PCA->2D then (angle bins + radius bins).
#       - better than angle-only for spiral/annulus-shaped clouds.
#
#     RANGE_zone:
#       - computed from Y by per-axis bins (physical excitation coverage).
#
#     Combining both avoids selecting many redundant points AND forces coverage of desired ranges.
#     """
#     if k_per_key is None:
#         return trials
#     k_per_key = int(k_per_key)
#     if k_per_key <= 0:
#         return []
#
#     X = np.asarray(X, float)
#     Y = np.asarray(Y, float)
#
#     if X.shape[0] != len(trials):
#         raise ValueError(f"X rows ({X.shape[0]}) must match trials ({len(trials)})")
#     if Y.shape[0] != len(trials):
#         raise ValueError(f"Y rows ({Y.shape[0]}) must match trials ({len(trials)})")
#
#     rng = np.random.default_rng(seed)
#
#     # --- PCA zone on X (angle+radius) ---
#     if zone_pca is None:
#         zone_pca = zone_id_from_X_pca_angle_radius(X, n_ang=pca_n_ang, n_rad=pca_n_rad)
#     zone_pca = np.asarray(zone_pca, dtype=int)
#     if zone_pca.shape[0] != len(trials):
#         raise ValueError("zone_pca must have shape (n_trials,) and match trials length")
#
#     # --- Range zone on Y ---
#     if edges_per_axis is None:
#         zone_ranges = np.zeros(len(trials), dtype=int)
#     else:
#         if zone_ranges is None:
#             zone_ranges = zone_id_from_Y_ranges(
#                 Y, edges_per_axis=edges_per_axis, axes=axes_for_ranges
#             )
#         zone_ranges = np.asarray(zone_ranges, dtype=int)
#         if zone_ranges.shape[0] != len(trials):
#             raise ValueError("zone_ranges must have shape (n_trials,) and match trials length")
#
#     # --- Bucket by (mass, pca_zone, range_zone) ---
#     buckets: dict[tuple[Any, int, int], list[int]] = {}
#     for i, t in enumerate(trials):
#         key = (mass_fn(t), int(zone_pca[i]), int(zone_ranges[i]))
#         buckets.setdefault(key, []).append(i)
#
#     kept: list[int] = []
#     for idxs in buckets.values():
#         idxs = np.asarray(idxs, dtype=int)
#         rng.shuffle(idxs)
#         kept.extend(idxs[: min(k_per_key, len(idxs))].tolist())
#
#     kept = sorted(set(kept))
#     return [trials[i] for i in kept]

def cap_trials_per_zone_mass_pca_and_ranges(
    trials: list[dict],
    X: np.ndarray,
    Y: np.ndarray,
    k_per_key: int,
    mass_fn: Callable[[dict], Any],
    seed: int = 0,
    pca_n_ang: int = 6,
    pca_n_rad: int = 3,
    edges_per_axis: Sequence[np.ndarray] | None = None,
    axes_for_ranges: Sequence[int] | None = None,
    zone_pca: np.ndarray | None = None,
    zone_ranges: np.ndarray | None = None,
    # NEW caps:
    max_per_pca_zone: int | None = None,
    max_total: int | None = None,
) -> list[dict]:
    if k_per_key is None:
        return trials
    k_per_key = int(k_per_key)
    if k_per_key <= 0:
        return []

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if X.shape[0] != len(trials):
        raise ValueError(f"X rows ({X.shape[0]}) must match trials ({len(trials)})")
    if Y.shape[0] != len(trials):
        raise ValueError(f"Y rows ({Y.shape[0]}) must match trials ({len(trials)})")

    rng = np.random.default_rng(seed)

    # --- PCA zones ---
    if zone_pca is None:
        zone_pca = zone_id_from_X_pca_angle_radius(X, n_ang=pca_n_ang, n_rad=pca_n_rad)
    zone_pca = np.asarray(zone_pca, dtype=int)
    if zone_pca.shape[0] != len(trials):
        raise ValueError("zone_pca must match trials length")

    # --- Range zones ---
    if edges_per_axis is None:
        zone_ranges = np.zeros(len(trials), dtype=int)
    else:
        if zone_ranges is None:
            zone_ranges = zone_id_from_Y_ranges(Y, edges_per_axis=edges_per_axis, axes=axes_for_ranges)
        zone_ranges = np.asarray(zone_ranges, dtype=int)
        if zone_ranges.shape[0] != len(trials):
            raise ValueError("zone_ranges must match trials length")

    # --- Stage 1: fine buckets (mass, pca, ranges) ---
    buckets: dict[tuple[Any, int, int], list[int]] = {}
    for i, t in enumerate(trials):
        key = (mass_fn(t), int(zone_pca[i]), int(zone_ranges[i]))
        buckets.setdefault(key, []).append(i)

    kept_idx: list[int] = []
    for idxs in buckets.values():
        idxs = np.asarray(idxs, dtype=int)
        rng.shuffle(idxs)
        kept_idx.extend(idxs[: min(k_per_key, len(idxs))].tolist())

    kept_idx = sorted(set(kept_idx))

    # --- Stage 2a: cap per PCA zone (ignoring mass/ranges) ---
    if max_per_pca_zone is not None:
        max_per_pca_zone = int(max_per_pca_zone)
        if max_per_pca_zone <= 0:
            return []

        by_zone: dict[int, list[int]] = {}
        for i in kept_idx:
            z = int(zone_pca[i])
            by_zone.setdefault(z, []).append(i)

        new_kept: list[int] = []
        for z, idxs in by_zone.items():
            idxs = np.asarray(idxs, dtype=int)
            rng.shuffle(idxs)
            new_kept.extend(idxs[: min(max_per_pca_zone, len(idxs))].tolist())

        kept_idx = sorted(set(new_kept))

    # --- Stage 2b: cap total ---
    if max_total is not None:
        max_total = int(max_total)
        if max_total <= 0:
            return []
        if len(kept_idx) > max_total:
            idxs = np.asarray(kept_idx, dtype=int)
            rng.shuffle(idxs)
            kept_idx = sorted(idxs[:max_total].tolist())

    return [trials[i] for i in kept_idx]

from typing import Callable, Any, Sequence
import numpy as np

AXES_6 = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

def make_edges_uniform(min_val: float, max_val: float, n_bins: int) -> np.ndarray:
    """Bins uniformes sur une plage physique (recommandé)."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    # n_bins bins => n_bins+1 edges
    return np.linspace(float(min_val), float(max_val), int(n_bins) + 1)
def zone_id_from_Y_ranges(
    Y: np.ndarray,
    edges_per_axis: Sequence[np.ndarray],
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    Y = np.asarray(Y, float)
    n = Y.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    if axes is None:
        axes = list(range(len(edges_per_axis)))
    if len(axes) != len(edges_per_axis):
        raise ValueError("axes and edges_per_axis must have the same length")

    per_axis_bins: list[np.ndarray] = []
    sizes: list[int] = []

    for ax, edges in zip(axes, edges_per_axis):
        edges = np.asarray(edges, float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Each edges array must be 1D with at least 2 values")

        z = np.digitize(Y[:, ax], edges, right=False) - 1
        z = np.clip(z, 0, edges.size - 2)
        per_axis_bins.append(z.astype(int))
        sizes.append(int(edges.size - 1))

    zone = np.zeros(n, dtype=int)
    mult = 1
    for z, s in zip(per_axis_bins, sizes):
        zone += z * mult
        mult *= s

    return zone
def make_edges_quantile(y: np.ndarray, n_bins: int) -> np.ndarray:
    """Bins par quantiles (utile si tu veux équilibrer les counts)."""
    y = np.asarray(y, float).ravel()
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(y, qs)
    edges = np.unique(edges)
    # fallback si trop de valeurs identiques
    if edges.size < 2:
        v = float(y[0]) if y.size else 0.0
        return np.array([v - 1.0, v + 1.0], float)
    return edges
def coverage_1d_counts(Y: np.ndarray, edges_per_axis) -> dict[str, np.ndarray]:
    """Retourne le comptage par bin sur chaque axe (diagnostic)."""
    Y = np.asarray(Y, float)
    out: dict[str, np.ndarray] = {}
    for ax, edges in enumerate(edges_per_axis):
        edges = np.asarray(edges, float)
        z = np.digitize(Y[:, ax], edges, right=False) - 1
        z = np.clip(z, 0, edges.size - 2)
        counts = np.bincount(z, minlength=edges.size - 1)
        out[AXES_6[ax]] = counts
    return out
def y_key_from_tolerances(Y: np.ndarray, tol: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, float)
    tol = np.asarray(tol, float).ravel()

    if tol.size != Y.shape[1]:
        raise ValueError("tol must match Y columns")
    if np.any(tol <= 0):
        raise ValueError("All tolerances must be > 0")

    Q = np.rint(Y / tol[None, :]).astype(np.int64)

    keys = [tuple(map(int, row)) for row in Q]
    return np.array(keys, dtype=object)
def y_key_from_relative_tolerances(
    Y: np.ndarray,
    pct: np.ndarray,        # ex: [0.05,...] (5%)
    min_tol: np.ndarray,    # plancher absolu par axe
) -> np.ndarray:
    Y = np.asarray(Y, float)
    pct = np.asarray(pct, float).ravel()
    min_tol = np.asarray(min_tol, float).ravel()

    if pct.size != Y.shape[1] or min_tol.size != Y.shape[1]:
        raise ValueError("pct and min_tol must match Y columns")
    if np.any(pct <= 0) or np.any(min_tol <= 0):
        raise ValueError("pct and min_tol must be > 0")

    tol_ij = np.maximum(min_tol[None, :], pct[None, :] * np.abs(Y))

    Q = np.rint(Y / tol_ij).astype(np.int64)
    keys = [tuple(map(int, row)) for row in Q]
    return np.array(keys, dtype=object)


def default_score_fn(X: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """
    Score "informatif" simple (sans PCA) :
    ||X|| sur X normalisé (z-score par colonne).
    """
    X = np.asarray(X, float)
    idxs = np.asarray(idxs, dtype=int)

    Xs = X.copy()
    mu = Xs.mean(axis=0, keepdims=True)
    sd = Xs.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Xs = (Xs - mu) / sd

    return np.linalg.norm(Xs[idxs], axis=1)
def counts_per_axis_bin(Y, edges_per_axis, axes=None):
    Y = np.asarray(Y, float)
    if axes is None:
        axes = list(range(len(edges_per_axis)))

    out = {}
    for ax, edges in zip(axes, edges_per_axis):
        edges = np.asarray(edges, float)
        z = np.digitize(Y[:, ax], edges, right=False) - 1
        z = np.clip(z, 0, len(edges) - 2).astype(int)

        counts = np.bincount(z, minlength=len(edges) - 1)
        out[ax] = counts  # counts[k] = nb essais dans le bin k
    return out
def cap_trials_per_zone_mass_and_Y_ranges_no_pca(
    trials: list[dict],
    X: np.ndarray,
    Y: np.ndarray,
    k_per_key: int | None,
    mass_fn: Callable[[dict], Any],
    seed: int = 0,
    edges_per_axis: Sequence[np.ndarray] | None = None,
    axes_for_ranges: Sequence[int] | None = None,
    zone_ranges: np.ndarray | None = None,
    pct: np.ndarray | None = None,
    min_tol: np.ndarray | None = None,
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = default_score_fn,
    quality_fn: Callable[[dict], float] | None = None,
    max_total: int | None = None,
) -> list[dict]:

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = len(trials)
    if X.shape[0] != n or Y.shape[0] != n:
        raise ValueError("X and Y must have same number of rows as trials")

    rng = np.random.default_rng(seed)

    # --- zones Y (ranges/bins) ---
    if edges_per_axis is None:
        zone_ranges = np.zeros(n, dtype=int)
    else:
        if zone_ranges is None:
            zone_ranges = zone_id_from_Y_ranges(Y, edges_per_axis=edges_per_axis, axes=axes_for_ranges)
        zone_ranges = np.asarray(zone_ranges, dtype=int)


        unique, counts = np.unique(zone_ranges, return_counts=True)
        order = np.argsort(counts)[::-1]  # tri décroissant

        for z, c in zip(unique[order], counts[order]):
            print(f"zone_id {z:6d} -> {c} essais")

        print("nb zones utilisées:", len(unique))
        print("total essais:", counts.sum())

        if zone_ranges.shape[0] != n:
            raise ValueError("zone_ranges must match trials length")


    if pct is not None:
        y_keys = y_key_from_relative_tolerances(Y, pct, min_tol)

        for i, k in enumerate(y_keys):
            print(f"{i:4d} -> {k}")
    else:
        y_keys = None

    # # --- buckets = (mass, zone_ranges) ---
    # buckets: dict[tuple[Any, int], list[int]] = {}
    # for i, t in enumerate(trials):
    #     key = (mass_fn(t), int(zone_ranges[i]))
    #     buckets.setdefault(key, []).append(i)
    #
    # print(buckets)
    #
    # kept_idx: list[int] = []
    #
    # for idxs in buckets.values():
    #     idxs = np.asarray(idxs, dtype=int)
    #     rng.shuffle(idxs)
    #
    #     # 1) dédoublonnage dans le bucket
    #     if y_keys is not None:
    #         groups: dict[Any, list[int]] = {}
    #         for i in idxs.tolist():
    #             k = tuple(np.asarray(y_keys[i]).ravel().tolist())
    #             groups.setdefault(k, []).append(i)
    #
    #         winners: list[int] = []
    #         for g in groups.values():
    #             g_idx = np.asarray(g, dtype=int)
    #
    #             scores = score_fn(X, g_idx)
    #
    #             if quality_fn is not None:
    #                 q = np.array([float(quality_fn(trials[i])) for i in g_idx], float)
    #                 # combine (zscore + zscore)
    #                 s = (scores - scores.mean()) / (scores.std() + 1e-12)
    #                 qn = (q - q.mean()) / (q.std() + 1e-12)
    #                 scores = s + qn
    #
    #             winners.append(int(g_idx[int(np.argmax(scores))]))
    #
    #         idxs2 = np.asarray(winners, dtype=int)
    #     else:
    #         idxs2 = idxs
    #
    #     # 2) cap par bucket
    #     if k_per_key is None:
    #         kept_idx.extend(idxs2.tolist())
    #     else:
    #         k = int(k_per_key)
    #         if k <= 0:
    #             continue
    #         rng.shuffle(idxs2)
    #         kept_idx.extend(idxs2[: min(k, len(idxs2))].tolist())
    #
    # kept_idx = sorted(set(kept_idx))
    #
    # # --- cap total ---
    # if max_total is not None:
    #     mt = int(max_total)
    #     if mt <= 0:
    #         return []
    #     if len(kept_idx) > mt:
    #         arr = np.asarray(kept_idx, dtype=int)
    #         rng.shuffle(arr)
    #         kept_idx = sorted(arr[:mt].tolist())
    #
    # return [trials[i] for i in kept_idx]

    # --- buckets = (zone_ranges) ---  (sans masse)
    buckets: dict[int, list[int]] = {}
    for i in range(len(trials)):
        key = int(zone_ranges[i])
        buckets.setdefault(key, []).append(i)

    print(buckets)

    kept_idx: list[int] = []

    for idxs in buckets.values():
        idxs = np.asarray(idxs, dtype=int)
        rng.shuffle(idxs)

        # 1) dédoublonnage dans le bucket
        if y_keys is not None:
            groups: dict[tuple[int, ...], list[int]] = {}
            for i in idxs.tolist():
                k = tuple(np.asarray(y_keys[i]).ravel().astype(int).tolist())
                groups.setdefault(k, []).append(i)

            winners: list[int] = []
            for g in groups.values():
                g_idx = np.asarray(g, dtype=int)

                scores = score_fn(X, g_idx)

                if quality_fn is not None:
                    q = np.array([float(quality_fn(trials[i])) for i in g_idx], float)
                    # combine (zscore + zscore)
                    s = (scores - scores.mean()) / (scores.std() + 1e-12)
                    qn = (q - q.mean()) / (q.std() + 1e-12)
                    scores = s + qn

                winners.append(int(g_idx[int(np.argmax(scores))]))

            idxs2 = np.asarray(winners, dtype=int)
        else:
            idxs2 = idxs

        # 2) cap par bucket
        if k_per_key is None:
            kept_idx.extend(idxs2.tolist())
        else:
            k = int(k_per_key)
            if k <= 0:
                continue
            rng.shuffle(idxs2)
            kept_idx.extend(idxs2[: min(k, len(idxs2))].tolist())

    kept_idx = sorted(set(kept_idx))

    # --- cap total ---
    if max_total is not None:
        mt = int(max_total)
        if mt <= 0:
            return []
        if len(kept_idx) > mt:
            arr = np.asarray(kept_idx, dtype=int)
            rng.shuffle(arr)
            kept_idx = sorted(arr[:mt].tolist())

    return [trials[i] for i in kept_idx]



